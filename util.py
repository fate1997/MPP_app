import streamlit as st

def formated_text(
    content,
    font_family='sans-serif',
    color='black',
    font_size='16px',
    font_weight='auto'
    ):
    text = f'<p style="color: {color}; font-weight: {font_weight}; font-size: {font_size};">{content}</p>'
    return text