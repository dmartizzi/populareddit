import os
import subprocess
from flask import render_template
from flask import request
from controversit_app import app
from controversit_app.utils import *
import pandas as pd

@app.route('/')
def root():
    return render_template("input.html")
    
@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/input')
def input():
    rpath = os.getcwd()
    static_path = rpath+"/controversit_app/static/images"
    subprocess.run(["rm",static_path+"/popularity_hist.png",static_path+"/controversiality_hist.png"])
    return render_template("input.html")

@app.route('/output')
def output():

  #pull 'subm_text' from input field and store it
  subm_text = request.args.get('subm_text')
  subr_name = request.args.get('subr_name')
  subr = subr_name.strip("r/")

  # Paths
  rpath = os.getcwd()
  nmf_model_path = rpath+"/controversit_app/nmf_models_prod"
  reg_model_path = rpath+"/controversit_app/regression_models_prod"
  pre_rendered_plots_path = rpath+"/controversit_app/pre_rendered_plots_prod"
  static_path = rpath+"/controversit_app/static/images"

  list_of_subr = get_list_of_subr(nmf_model_path)

  subprocess.run(["rm",static_path+"/popularity_hist.png",static_path+"/controversiality_hist.png"])
  subprocess.run(["cp",pre_rendered_plots_path+"/popularity_hist_r-"+subr+".png",static_path+"/popularity_hist.png"],stdout=subprocess.PIPE, stderr=subprocess.STDOUT)  
  subprocess.run(["cp",pre_rendered_plots_path+"/controversiality_hist_r-"+subr+".png",static_path+"/controversiality_hist.png"],stdout=subprocess.PIPE, stderr=subprocess.STDOUT)  
  
  nmf_model,success_nmf,vectorizer,success_vec,transformer,success_tra = load_nmf_model(nmf_model_path,subr)

  try:
     temp = nmf_model.components_
     num_topics = temp.shape[0]
  except:
     num_topics=0

  reg_pop_model,score_pop,success_pop,reg_con_model, \
      score_con,success_con, \
      pop_fraction,con_fraction = load_reg_model(reg_model_path,subr)
  
  success = success_nmf and success_vec and success_tra and success_pop and success_con
 
  pop_mess = "UNPOPULAR"
  pop_color = "red"
  score_pop_s = ""
  con_mess = "UNCONTROVERSIAL"
  con_color = "green" 
  score_con_s = ""
  if success == True:
     if len(subm_text.split())<=3 :
        template_out = "output_error.html"
        message = "Error! The submission is too short to make a prediction. Please, submit a longer text..."
        message_color = "red"
     else :        
        template_out = "output.html"
        message = "Prediction performed!"
        message_color = "green"
        popular,controversial = run_full_model(subm_text,nmf_model,vectorizer,transformer,reg_pop_model,reg_con_model)
        if int(popular[0]) == 1:
           pop_mess = "POPULAR"
           pop_color = "green"
        if int(controversial[0]) == 1:
           con_mess = "CONTROVERSIAL"
           con_color = "red"

        if score_pop < 55:
            score_pop_s = "LOW"
        if score_pop >= 55 and score_pop <= 85:
            score_pop_s = "MODERATE"
        if score_pop > 85:
            score_pop_s = "HIGH"

        if score_con < 55:
            score_con_s = "LOW"
        if score_con >= 55 and score_con <= 85:
            score_con_s = "MODERATE"
        if score_con > 85:
            score_con_s = "HIGH"

  else :
     template_out = "output_error.html"
     message = "Error! Subreddit "+subr_name+" is not in the database. Please, choose another subreddit..."
     message_color = "red"

  return render_template(template_out,subr_name=subr_name, \
                         subm_text=subm_text,message=message, \
                         message_color=message_color, \
                         pop_mess=pop_mess,pop_color=pop_color,score_pop=score_pop_s, \
                         con_mess=con_mess,con_color=con_color,score_con=score_con_s, \
                         pop_fraction=pop_fraction,con_fraction=con_fraction)
