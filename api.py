import pandas as pd

# Initialize a dictionary
dict = {'Students':['Harry', 'John', 'Hussain', 'Satish'],
        'Scores':[77, 59, 88, 93]}

# Create a DataFrame
df = pd.DataFrame(dict)

# Save DataFrame as JSON File in same folder in which code is running
df.to_json('api.json', orient='index')