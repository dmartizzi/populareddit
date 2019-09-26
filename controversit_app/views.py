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
    subprocess.run(["rm",static_path+"/popularity_hist.png"])
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

  # List of Subreddits
  list_of_subr = get_list_of_subr(nmf_model_path)
  
  # Statif Figures - Histograms
  setup_static_figures(pre_rendered_plots_path,static_path,subr)

  # NMF Topic model for this subreddit
  nmf_model,success_nmf,vectorizer,success_vec,transformer,success_tra = load_nmf_model(nmf_model_path,subr)

  try:
     temp = nmf_model.components_
     num_topics = temp.shape[0]
  except:
     num_topics=0
  topics = ""

  # Regression model for this subreddit
  reg_pop_model,score_pop,success_pop,pop_fraction = load_reg_model(reg_model_path,subr)

  success = success_nmf and success_vec and success_tra and success_pop
 
  pop_mess = "UNPOPULAR"
  pop_color = "red"
  score_pop_s = ""
  num_top_words = 0
  alt_subr_count = 1
  alt_subr = ["Unable to recommend alternative subreddits."]
  if success == True:
     if len(subm_text.split())<=3 :
        template_out = "output_error.html"
        message = "Error! The submission is too short to make a prediction. Please, submit a longer text..."
        message_color = "red"
     else :        
        template_out = "output.html"
        message = "Prediction performed!"
        message_color = "green"
        popular = run_full_model(subm_text,nmf_model,vectorizer,transformer,reg_pop_model)
        if int(popular[0]) == 1:
           pop_mess = "POPULAR"
           pop_color = "green"

        if score_pop < 55:
            score_pop_s = "LOW"
        if score_pop >= 55 and score_pop <= 85:
            score_pop_s = "MODERATE"
        if score_pop > 85:
            score_pop_s = "HIGH"

        num_top_words = 20
        sorter = np.argsort(-reg_pop_model.feature_importances_)
        topics = get_nmf_topics(vectorizer,nmf_model,sorter,num_top_words,num_topics)

        maxalt = 5
        alt_subr,alt_subr_count = find_alternative_subreddits(subr,list_of_subr,subm_text,maxalt)
        
  else :
     template_out = "output_error.html"
     message = "Error! Subreddit "+subr_name+" is not in the database. Please, choose another subreddit..."
     message_color = "red"

  return render_template(template_out,subr_name=subr_name, \
                         subm_text=subm_text,message=message, \
                         message_color=message_color, \
                         pop_mess=pop_mess,pop_color=pop_color,score_pop=score_pop_s, \
                         pop_fraction=pop_fraction,num_top_words=num_top_words, \
                         topics=topics,alt_subr=alt_subr,alt_subr_count=alt_subr_count)
