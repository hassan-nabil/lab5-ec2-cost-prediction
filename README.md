# Lab 5 — EC2 Cost Prediction Dashboard

This Windows-policy-friendly, deployment-ready app uses `ec2dataset.csv` from the earlier EC2 analysis activity. The original dataset contains **861 rows and 11 columns**. It predicts Linux EC2 On-Demand **hourly USD cost** using vCPUs and memory (GiB).

## Windows Application Control compatibility

This edition **does not use scikit-learn or SciPy**. It computes ordinary least-squares
regression using NumPy, so the SciPy `rcont` DLL blocked by some Windows Application
Control policies is never loaded. It retains the same course concepts (80/20 split,
linear regression, MAE, RMSE, R², and visual comparison). **No changes to Windows
security settings are needed.**

## Run locally on Windows

Extract the ZIP and open the **Lab5_EC2_Cost_Prediction** folder in VS Code. You must run these commands from the folder containing `app.py` and `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

This runs without activating PowerShell scripts, so the PowerShell ExecutionPolicy error will not interfere. Open http://localhost:8501 if the browser does not open automatically.

## Deploy to Streamlit Community Cloud (MANDATORY)

1. Sign in to your own GitHub account at https://github.com/.
2. Create a new repository, e.g. `lab5-ec2-prediction` (public is simplest).
3. Upload these three files into the **root** of the repository: `app.py`, `ec2dataset.csv`, `requirements.txt`. You may also upload `README.md`.
   **Do not upload the outer ZIP, `.venv` folder, or just a nested project directory.**
4. Commit the files to the default branch (usually `main`).
5. Visit https://share.streamlit.io/ and sign in/connect GitHub.
6. Click **Create app** → **Yup, I have an app**.
7. Choose the GitHub repository, branch `main`, and **Main file path** `app.py`.
8. Click **Deploy**. Wait for dependency installation and startup.
9. Open the public `*.streamlit.app` link and test that prediction and the charts work.
10. Submit/share your live app URL and prepare to demo it.

Official docs: https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started

## Two-minute demo script

**1. Overview (15 seconds)**

> This is my Lab 5 EC2 instance cost prediction dashboard. It uses the 861-row EC2 dataset from the previous activity, cleans the numeric fields, and trains regression models to estimate On-Demand Linux pricing using memory and vCPUs.

**2. Predict (30 seconds)**

> I can enter a hypothetical instance like 4 GiB memory and 2 vCPUs. The dashboard estimates hourly and monthly costs. It also shows actual priced EC2 instance types with matching specifications, since a model estimate is not the actual AWS price.

**3. Validate (30 seconds)**

> I trained using 80% of valid data and tested on the other 20%. This chart compares actual prices with predicted prices. MAE and RMSE measure the model's error, and the red dashed line is perfect prediction.

**4. Explain model limitation (20 seconds)**

> The activity's basic linear regression can predict negative prices for small instances, which are impossible. I kept that baseline for comparison and added an optional log-linear regression, which generates positive estimates. Both use ordinary least-squares regression, but the log-linear version fits the logarithms of features and price.

**5. Optimization (20 seconds)**

> The shortlist finds instances that meet the requested CPU and memory at the lowest listed prices. I would choose based on performance needs and actual AWS regional prices, then compare On-Demand, Reserved and Spot for appropriate workloads.

**6. Cloud deployment (10 seconds)**

> I deployed the dashboard on Streamlit Community Cloud through GitHub, so anyone with the public link can test it.

## What to say if asked about results

- The basic linear-regression model directly predicts hourly price from numeric RAM and vCPU fields.
- The dataset has **20 rows without valid On-Demand pricing**, leaving **841 priced instances**, with **672 training** and **169 testing** observations at an 80/20 split.
- Predictions are **estimates**, not live quotes. Prices vary by region, instance generation, networking, architecture and pricing plan.
- The baseline can have good overall R² while still giving unreasonable predictions for low-cost instances. That is a legitimate model limitation, not a software error.
- Optimizing means meeting workload requirements at a lower cost, not blindly selecting the globally cheapest instance.

## Notes

This repository contains a teacher-provided EC2 pricing snapshot, not credentials, and needs no cloud secrets or paid AWS account. Streamlit Community Cloud needs your own GitHub/Streamlit login for deployment.
