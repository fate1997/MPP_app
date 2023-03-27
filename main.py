import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu

import style
from page.home import home
from page.organic_liquid import OrganicLiquid
from page.organic_gas import OrganicGas


# multiple pages implementation
class MultiApp:
    def __init__(self):
        self.apps = []
        self.app_dict = {}

    def add_app(self, title, func):
        if title not in self.apps:
            self.apps.append(title)
            self.app_dict[title] = func

    
    def run(self):
        image = Image.open('./image/icon.png')
        st.sidebar.image(image, use_column_width=True)
        with st.sidebar:
            st.text('')
            app = option_menu("", 
                            self.apps, 
                            icons=['house', 'list-task', 'list-task'], 
                            menu_icon="", 
                            default_index=0,
                            styles=style.menu)
        self.app_dict[app]()


if __name__ == '__main__':
    app = MultiApp()
    app.add_app("Home", home)
    
    organic_liquid = OrganicLiquid()
    app.add_app("General Organic Liquid", organic_liquid.show)
    app.add_app("General Organic Gas", OrganicGas().show)
    app.run()