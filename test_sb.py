import os
from supabase import create_client

_SB_URL = "https://ujlqigpwoaewdrylesxg.supabase.co"
_SB_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVqbHFpZ3B3b2Fld2RyeWxlc3hnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzc1MTk4NDcsImV4cCI6MjA5MzA5NTg0N30.zmcXsds_nFWjpPdh9V_RtpMV3HNrR4Dgk9VsuZ8Jd8Y"
_sb = create_client(_SB_URL, _SB_KEY)

try:
    print(dir(_sb.auth))
except Exception as e:
    print(e)
