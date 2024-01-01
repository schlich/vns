from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as vuetify

server = get_server(client_type="vue3")
state, ctrl = server.state, server.controller


with SinglePageLayout(server) as layout:
    with layout.toolbar as toolbar:
        toolbar.dense = True  # Update toolbar attribute
        vuetify.VSpacer()  # Push things to the right
        vuetify.VSlider(  # Add slider
            v_model=("resolution", 6),  # bind variable with an initial value of 6
            min=3,
            max=60,  # slider range
            dense=True,
            hide_details=True,  # presentation setup
        )
        # Bind methods to 2 icon buttons
        with vuetify.VBtn(icon=True):
            vuetify.VIcon("mdi-crop-free")
        with vuetify.VBtn(icon=True):
            vuetify.VIcon("mdi-undo")
    with layout.content, vuetify.VContainer(fluid=True, classes="pa-0 fill-height"):
        ...

if __name__ == "__main__":
    server.start()
