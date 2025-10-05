# python_lab_NLP

<H3>Project Files Installation:</H3>

1. Recommend creating a virtual environment with python 3.12.2 using the command:<br>
    <code>virtualenv -p <interpreter-path(python.exe path)> <my_env_name></code>
  
2. Activate the virtualenv:<br>
    <code>source <my_env_name>/bin/activate</code>
    or
   <code><my_env_name>/Scripts/activate</code>
  
4. Use pip/pip3 and requirements.txt to install required packages:<br>
    <code>pip install -r requirements.txt</code>
  
5. Manually install other missing packages using pip command:<br>
    <code> pip install <module> </code> 
<h3>Project Directory after successful installation(Virtual environment name = env):</h3>        
<br><img src="Repository_extra/Post Installation.JPG.png">

6. Run <code> train_models.py </code> to train and initialise the models.<br>
    <code> python train_models.py </code>
   
7. Edit app.py as required
  
8. Run the streamlit webserver using the commmand:<br>
    <code>streamlit run app.py</code>
