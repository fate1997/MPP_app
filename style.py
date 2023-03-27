import streamlit as st

# menu
menu = {"container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "#FF6666", "font-size": "16px", "font-family": "Consola"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"10px", "--hover-color": "#0099CC"},
        "nav-link-selected": {"background-color": "#0099CC", "font-weight": "inherit"}, }

text =  """
        <style>
        textarea {
        font-size: 1.5rem !important;
        }
        input {
        font-size: 1.5rem !important;
        }
        </style>
        """
        
button = """
        <style>
        div.stButton > button:first-child {
            background-color: #0099CC;
            color: white;
            height: 3em;
            width: 15em;
            border-radius:10px;
            border:3px solid #CCCCCC;
            font-size:18px;
            font-weight: bold;
            margin: auto;
            display: block;
        }

        div.stButton > button:hover {
            background:linear-gradient(to bottom, #0099CC 5%, #0066CC 100%);
            background-color:#0099CC;
        }

        div.stButton > button:active {
            position:relative;
            top:1px;
        }
        
        </style>"""              