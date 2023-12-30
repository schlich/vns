import logging

from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vtk
from trame.widgets import vuetify3 as vuetify

server = get_server()

server.client_type = "vue3"

state, ctrl = server.state, server.controller


def reset_resolution():
    state.resolution = 6


# When resolution change, execute fn
@state.change("resolution")
def resolution_change(resolution):
    logging.info("Slider updating resolution to ", extra={resolution: resolution})


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
        with vuetify.VBtn(icon=True, click=ctrl.reset_camera):
            vuetify.VIcon("mdi-crop-free")
        with vuetify.VBtn(icon=True, click=reset_resolution):
            vuetify.VIcon("mdi-undo")
    with layout.content, vuetify.VContainer(fluid=True), vtk.VtkView() as vtk_view:
        ctrl.reset_camera = vtk_view.reset_camera
        with vtk.VtkGeometryRepresentation():
            vtk.VtkAlgorithm(
                vtkClass="vtkConeSource",
                state=("{ resolution }",),
            )


server.start()
