import streamlit as st
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
from streamlit_option_menu import option_menu


import streamlit as st

class MultiApp:
    def __init__(self):
        self.apps = []
        self.app_dict = {}

    def add_app(self, title, func):
        if title not in self.apps:
            self.apps.append(title)
            self.app_dict[title] = func

    def run(self):
        image = Image.open('./images/title_nobg.png')
        st.sidebar.image(image, use_column_width=True)
        app = st.sidebar.radio(
            '',
            self.apps,
            format_func=lambda title: str(title))

        self.app_dict[app]()
    
    def run_menu(self):
        image = Image.open('./images/icon.png')
        st.sidebar.image(image, use_column_width=True)
        with st.sidebar:
            st.text('')
            app = option_menu("", 
                            self.apps, 
                            icons=['house', 'list-task', 'list-task'], 
                            menu_icon="", 
                            default_index=0,
                            styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "#FF6666", "font-size": "16px", "font-family": "Consola"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"10px", "--hover-color": "#0099CC"},
        "nav-link-selected": {"background-color": "#0099CC", "font-weight": "inherit"},
    })
        self.app_dict[app]()



def organic_molecule():
    
    image = Image.open('./images/organic_title.png')
    st.image(image, use_column_width=True)

    col1, col2 = st.columns(spec=(2, 1))
    # Store the initial value of widgets in session state
    with col1:
        input_molecule_header = st.subheader("INPUT MOLECULE")
        smiles = st.text_input('Enter molecular ID in format of CAS or Name or SMILES', placeholder="64-17-5/ethanol/CCO")

    with col2:
        input_task_header = st.subheader("TEMPERATURE")
        temperature = st.text_input('Enter temperature with a unit of Kelvin', placeholder="273.15")

    st.markdown(
            """
            <style>
            textarea {
                font-size: 1.5rem !important;
            }
            input {
                font-size: 1.5rem !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        ) 
    
    st.subheader('PROPERTY')
    st.markdown('#### Temperature-related')
    c1, c2, c3= st.columns(3)
    with c1:    visc = st.checkbox("Viscosity")
    with c2:    thermal_cond = st.checkbox("Thermal conductivity")
    with c3:    diffusion_coef = st.checkbox("Diffusion coefficient")
    st.markdown('#### Temperature-unrelated')
    c1, c2, c3= st.columns(3)
    with c1:    critical = st.checkbox("Critical properties")
    # with c2:    thermal_cond = st.checkbox("Thermal conductivity")
    # with c3:    diffusion_coef = st.checkbox("Diffusion coefficient")
    
    m = st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #0099CC;
            color: white;
            height: 3em;
            width: 15em;
            border-radius:10px;
            border:3px solid #CCCCCC;
            font-size:18px;
            font-weight: auto;
            margin: auto;
            display: block;
        }

        div.stButton > button:hover {
            background:linear-gradient(to bottom, #0099CC 5%, #0066CC 100%);
            background-color:#0099CC;
        }

        div.stButton > button:active {
            position:relative;
            top:3px;
        }

        </style>""", unsafe_allow_html=True)
    run = st.button('RUN')
    
    st.subheader('OUTPUT')
    col1, col2, col3 = st.columns(3)
    with col1:
        new_title = '<p style="font-family:sans-serif; ; font-size: 20px;">Property Name</p>'
        st.markdown(new_title, unsafe_allow_html=True)
    with col2:
        new_title = '<p style="font-family:sans-serif; ; font-size: 20px;">Property value</p>'
        st.markdown(new_title, unsafe_allow_html=True)
    with col3:
        new_title = '<p style="font-family:sans-serif; ; font-size: 20px;">Unit</p>'
        st.markdown(new_title, unsafe_allow_html=True)
    if run:
        with col1:
            new_title = '<p style="font-family:sans-serif; color: red; font-size: 18px;">viscosity</p>'
            if visc: st.markdown(new_title, unsafe_allow_html=True)
            
        with col2:
            value = 3
            if visc: st.markdown(f'**{value}**')

        
        with col3:
            if visc: st.markdown('**mPa·s**')

    

def deep_eutectic_solvents():
    ##################
    # Input Text Box
    ##################
    image = Image.open('./images/des_title.png')
    st.image(image, use_column_width=True)
    col1, col2 = st.columns(spec=(2, 1))
    # Store the initial value of widgets in session state
    with col1:
        input_molecule_header = st.subheader("INPUT DES")
        smiles = st.text_input('Enter your molecular ID in format of CAS or Name or SMILES', placeholder="64-17-5/ethanol/CCO")

    with col2:
        input_task_header = st.subheader("PROPERTY")
        task = st.selectbox('What property you want to predict', 
                            ("Viscosity", "Thermal Conductivity", "Diffusion Coefficient", "Diffusion Coefficient in Water",
                            "Density", "Antoine Parameters"))

    ##################
    # Output Text Box
    ##################
    st.subheader('OUTPUT')
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write('Molecule')
        mol = Chem.MolFromSmiles(smiles)
        im = Draw.MolToImage(mol)
        st.image(im, width=50)

    with col2:
        st.write('Viscosity')
        st.write(1.02)

    with col3:
        st.write('Unit')
        st.write('mPa·s')


def home():
    image = Image.open('./images/title2.jpg')
    st.image(image, use_column_width=True)
    st.markdown("## Aiming to readily access molecular property.")

app = MultiApp()
app.add_app("Home", home)
app.add_app("General Organic Matter", organic_molecule)
app.add_app("Deep Eutectic Solvent", deep_eutectic_solvents)
app.run_menu()