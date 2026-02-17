# stopwords.py

import nltk
from nltk.corpus import stopwords

# --- Filipino Stopwords (Optimized + Time Words) ---
filipino_stopwords = [
    # Markers & Pronouns
    "ang", "ng", "sa", "si", "ni", "kay", "ay", "na", "pa", "po", "ho", 
    "ako", "ikaw", "siya", "kami", "tayo", "kayo", "sila", 
    "ko", "mo", "niya", "namin", "natin", "ninyo", "nila",
    "akin", "iyo", "kanya", "amin", "atin", "inyo", "kanila",
    "aking", "iyong", "kanyang", "aming", "ating", "inyong", "kanilang",
    
    # Prepositions & Demonstratives
    "dito", "doon", "diyan", "d'yan", "iyan", "iyon", "ito",
    "nito", "niyan", "niyon", "ganito", "ganyan", "ganoon",
    "mula", "tungo", "para", "tungkol", "laban", "bago", "habang",
    
    # Conjunctions & Connectors
    "at", "o", "pero", "ngunit", "subalit", "dahil", "kasi", "kung", 
    "kapag", "upang", "nang", "kaya", "sapagkat", "gayunman", "samantala",
    
    # Common Verbs/Adverbs & Negation
    "may", "mayroon", "meron", "wala", "walang", "hindi", "huwag", "di", "wag",
    "dapat", "kailangan", "gusto", "nais", "ibig", "maaari", "pwede",
    "sabi", "sabihin", "sinabi", "daw", "raw",
    
    # Numbers & Quantifiers
    "isa", "dalawa", "tatlo", "apat", "lima",
    "bawat", "lahat", "marami", "onti", "kaunti", "ilan",
    
    # Generic Time/State words (NEWLY ADDED)
    "ngayon", "bukas", "kahapon", "noon", "palagi", "minsan",
    "muli", "una", "pangalawa", "huli", "tapos", "pagkatapos",
    "maging", "naging", "pagkakaroon", "ginagawa", "ginawa", "gagawin",
    "araw", "buwan", "taon", "oras", "minuto", "segundo"
]

# --- English Stopwords + Time Words ---
try:
    english_stopwords = list(stopwords.words('english'))
    # --- ADD THIS BLOCK ---
    time_words = [
        "day", "days", "month", "months", "year", "years", "week", "weeks",
        "today", "yesterday", "tomorrow", "ago", "daily", "time", "hour", "hours"
    ]
    english_stopwords.extend(time_words)
    # ----------------------
except LookupError:
    print("NLTK stopwords not found. Downloading...")
    nltk.download('stopwords')
    english_stopwords = list(stopwords.words('english'))

# --- Combined Stopwords ---
combined_stopwords = list(set(english_stopwords + filipino_stopwords))