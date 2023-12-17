# import streamlit as st
# import pandas as pd
# import time
# from datetime import datetime

# ts = time.time()
# date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
# timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

# # Placeholder to refresh content
# placeholder = st.empty()

# count = 0

# while count < 100:
#     # Update count
#     count += 1

#     if count % 3 == 0 and count % 5 == 0:
#         result = "FizzBuzz"
#     elif count % 3 == 0:
#         result = "Fizz"
#     elif count % 5 == 0:
#         result = "Buzz"
#     else:
#         result = f"Count: {count}"

#     # Update content within the placeholder
#     placeholder.text(result)

#     # Simulate some delay
#     time.sleep(2)

# # Read CSV file
# df = pd.read_csv("recognition_log" + date + ".csv")

# # Display dataframe
# st.dataframe(df.style.highlight_max(axis=0))


import streamlit as st
import pandas as pd

# Load the CSV file
csv_file_path = 'recognition_log.csv'
df = pd.read_csv(csv_file_path)

# Streamlit app
st.title('Face Recognition Log Viewer')

# Display the dataframe
st.write(df)
