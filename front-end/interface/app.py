import streamlit as st
import requests

def run_app():
    st.title('Recommendation system')

    if st.button('Load data via API'):
        #st.write('Button')
        with st.spinner('Loading data... Please wait'):
            try:
                call = requests.post('http://localhost:8000/api/ingestion/run')
                if call.status_code == 200:
                    st.success("Data ingested successfully")
                    st.json(call.json())
                else:
                    st.error('Error API' + str(call.status_code))
                    st.error(call.text)
            except requests.exceptions.ConnectionError:
                st.error('Check that FastAPI is running.')
            except Exception as f:
                st.error('Unexpected error')
