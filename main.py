# main.py
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Annotated, Any
import operator
from dotenv import load_dotenv
import re

load_dotenv()

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Medical data processing
import pydicom
from PIL import Image
import tqdm
