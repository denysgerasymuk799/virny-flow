import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional, List, Dict, Any


class MongoDBClient:
    """MongoDB client for handling database operations"""
    
    def __init__(self, env_file_path: Optional[str] = None):
        """
        Initialize MongoDB client
        
        Args:
            env_file_path: Path to the environment file. If None, will look for secrets_gmail.env
        """
        self.client = None
        self.db = None
        self.connection_string = None
        self.db_name = None
        
        # Load environment variables
        if env_file_path is None:
            # Default path to secrets file
            env_file_path = str(Path(__file__).parent.parent.parent.joinpath('scripts').joinpath('configs').joinpath('secrets_gmail.env'))
        
        self._load_environment(env_file_path)
        self._connect()
    
    def _load_environment(self, env_file_path: str) -> None:
        """Load environment variables from the specified file"""
        try:
            load_dotenv(env_file_path)
            self.connection_string = os.getenv('CONNECTION_STRING')
            self.db_name = os.getenv('DB_NAME')
            if not self.connection_string or not self.db_name:
                raise ValueError("Missing MongoDB connection details in environment file")
            if self.db_name is None:
                raise ValueError("DB_NAME environment variable is not set.")
        except Exception as e:
            raise ConnectionError(f"Failed to load environment variables: {str(e)}")
    
    def _connect(self) -> None:
        """Establish connection to MongoDB"""
        try:
            self.client = MongoClient(self.connection_string)
            if self.db_name is None:
                raise ValueError("DB_NAME is not set. Cannot connect to database.")
            self.db = self.client[self.db_name]
            # Test the connection
            self.client.admin.command('ping')
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MongoDB: {str(e)}")
    
    def get_logical_pipeline_scores(self, exp_config_name: str = 'gradio_demo', run_num: int = 1) -> pd.DataFrame:
        """
        Get logical pipeline scores filtered by experiment configuration name
        
        Args:
            exp_config_name: Name of the experiment configuration to filter by
            
        Returns:
            pandas.DataFrame: DataFrame containing the pipeline scores
        """
        try:
            if self.db is None:
                raise ConnectionError("Database connection is not established.")
            collection = self.db['logical_pipeline_scores']
            
            # Query data filtered by exp_config_name
            query = {'exp_config_name': exp_config_name, 'run_num': run_num}
            cursor = collection.find(query)
            
            # Convert to DataFrame
            df = pd.DataFrame(list(cursor))
            
            # Drop MongoDB's _id column if it exists
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            if df.empty:
                return pd.DataFrame({'Message': [f'No data found for exp_config_name: {exp_config_name}']})
            
            return df
            
        except Exception as e:
            return pd.DataFrame({'Error': [f'Database query error: {str(e)}']})
    
    def get_all_collections(self) -> List[str]:
        """Get list of all collection names in the database"""
        try:
            if self.db is None:
                raise ConnectionError("Database connection is not established.")
            return self.db.list_collection_names()
        except Exception as e:
            raise Exception(f"Failed to get collections: {str(e)}")
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a specific collection"""
        try:
            if self.db is None:
                raise ConnectionError("Database connection is not established.")
            collection = self.db[collection_name]
            return {
                'count': collection.count_documents({}),
                'name': collection_name
            }
        except Exception as e:
            raise Exception(f"Failed to get collection stats: {str(e)}")
    
    def get_all_exp_config_names(self) -> List[str]:
        """
        Fetch all unique exp_config_name values from the exp_config_history collection.
        Returns:
            List[str]: List of unique exp_config_name values.
        """
        try:
            if self.db is None:
                raise ConnectionError("Database connection is not established.")
            collection = self.db['exp_config_history']
            exp_config_names = collection.distinct('exp_config_name')
            return sorted(exp_config_names)
        except Exception as e:
            raise Exception(f"Failed to fetch exp_config_name values: {str(e)}")
    
    def close(self) -> None:
        """Close the MongoDB connection"""
        if self.client:
            self.client.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
