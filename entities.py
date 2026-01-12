from typing import List, Union, Optional
from pydantic import BaseModel

class Tour(BaseModel):
    tour_name: str
    summary: str
    location: str
    duration: str
    price: str
    includes: List[str]
    notes: str
    style: str
    transport: Union[str, List[str]]
    accommodation: Union[str, List[str]]
    meals: str
    event_support: str

class KnowledgeBase(BaseModel):
    about_company: Optional[dict] = None
    tours: List[Tour]
