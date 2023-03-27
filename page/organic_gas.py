import streamlit as st
from PIL import Image

import style
from util import formated_text

from package.MolProp.MolProp_model import MolPropModel
import cirpy

from package.MolProp.app_utils import smiles2Data, load_config
import torch
import numpy as np

class OrganicGas:
    def __init__(self,  
                 task_unit={"Viscosity (G)": "$\\rm{µP}$", 
                            "Thermal conductivity (G)": "$\\rm{W/(m·K)}$",  
                            "Diffusion coefficient in air": "$\\rm{cm^2/s}$"}):
        self.tasks = task_unit.keys()
        self.task_unit = task_unit
        
        self.models = {}
        config = load_config('./package/MolProp/model_files/train.yml')
        for task in self.tasks:
            self.models[task] = MolPropModel(config).load_info('./package/MolProp/model_files/'+task+'.pt')
        for task in self.tasks:
            self.models[task].eval()
        
        
    def show(self):
        image = Image.open('./image/organic_gas.png')
        st.image(image, use_column_width=True)
        
        smiles, temperature, todo_tasks = self.inpt()
        run = self.outpt()
        
        if run:
            self.compute(smiles, temperature, todo_tasks)
        
    
    def inpt(self):
        
        # Get organic ID and temperature
        col1, col2 = st.columns(spec=(2, 1))
        with col1:
            st.subheader("INPUT MOLECULE")
            smiles = st.text_input('Enter molecular ID in format of CAS or Name or SMILES', placeholder="74-84-0/ethane/CC")
            
        with col2:
            st.subheader("TEMPERATURE")
            temperature = st.text_input('Enter temperature with a unit of Kelvin', placeholder="298.15")

        st.markdown(style.text, unsafe_allow_html=True) 
        
        # Get task
        st.subheader('PROPERTY')
        col1, col2, col3= st.columns(spec=(1, 1, 1.2))
        todo_tasks = {}
        for i, task in enumerate(self.tasks):
            if (i+1)%3 == 1:    todo_tasks[task] = col1.checkbox(task)
            if (i+1)%3 == 2:    todo_tasks[task] = col2.checkbox(task)
            if (i+1)%3 == 0:    todo_tasks[task] = col3.checkbox(task)
        
        return smiles, temperature, todo_tasks

    def outpt(self):
        st.markdown(style.button, unsafe_allow_html=True) 
        run = st.button('RUN')
        
        st.subheader('OUTPUT')
        
        # column name
        col1, col2, col3 = st.columns(3)

        col1.markdown('**Property Name**')
        col2.markdown('**Property value**')
        col3.markdown('**Unit**')
        
        return run

    
    def compute(self, smiles, temperature, todo_tasks):
        
        flag = 1
        # raise error if input is not complete
        if not smiles:
            st.error("Please enter a molecular ID.")
            flag = 0
        if not temperature:
            st.error("Please enter a temperature.")
            flag = 0
        try:
            temperature = float(temperature)
        except ValueError:
            st.error("Please ensure your temperatue is digit.")
            flag = 0
        
        if flag == 0:   return
        
        # load data
        smiles = cirpy.resolve(smiles, 'smiles')
        if not smiles:
            st.error("Please re-check the molecular ID.")
            flag = 0
        if flag == 0:   return
                    
        data = smiles2Data(smiles, temperature)
        
        task_value = {}
        for task in self.tasks:
            if todo_tasks[task]:        
                task_value[task] = f'{torch.exp(self.models[task](data)).item(): .3f}'
                
        
        col1, col2, col3 = st.columns(3)
        for task in task_value:
            col1.markdown(formated_text(task, color='#FF6666', font_size=20, font_weight='bold'), unsafe_allow_html=True)
            col2.markdown(formated_text(task_value[task], color='#FF6666', font_size=20, font_weight='bold'), unsafe_allow_html=True)
            col3.markdown(self.task_unit[task], unsafe_allow_html=True)
        return task_value