from roboflow import Roboflow
rf = Roboflow(api_key="8rqVWAx0AL7QhfeujEkJ")
project = rf.workspace("fyp-scksi").project("building-detection-from-satellite-images-ghtvy")
version = project.version(25)
dataset = version.download("coco")


