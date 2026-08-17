from arduino.app_utils import App

# The signtalk-stack custom Docker Brick is started by App Lab from app.yaml.
# Keep this required entry point alive while its NestJS and Edge AI services run.
App.loop()
