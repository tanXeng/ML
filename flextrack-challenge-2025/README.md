# 🌟 [FlexTrack: Detecting Energy Flexibility in Buildings](https://www.aicrowd.com/challenges/flextrack-challenge-2025) 🌟 Starter Kit


A global challenge to detect and quantify energy flexibility in buildings — enabling smarter demand response and accelerating the transition to a sustainable energy future.

Buildings, HVAC systems, batteries, and solar inverters can all flex their energy usage to support the grid. Detecting and quantifying these demand response events is critical for grid operators, aggregators, and facility managers to unlock flexibility markets.

This challenge invites you to develop models that can:

1. **Detect demand response activation**, and
2. **Measure demand response capacity delivered**.

📄 Register here: [https://www.aicrowd.com/challenges/flextrack-challenge-2025](https://www.aicrowd.com/challenges/flextrack-challenge-2025)
📧 Contact: Dr. Emily Yap ([eyap@uow.edu.au](mailto:eyap@uow.edu.au))

---

# Starter Kit

To get started, follow these easy steps:

1. **Join the competition**: Register your interest at [https://www.aicrowd.com/challenges/flextrack-challenge-2025](https://www.aicrowd.com/challenges/flextrack-challenge-2025)
2. Click on the **Participate** button, and accept the Rules of the competition. 
3. **Download the dataset** from the Resources section of the competition page. [https://www.aicrowd.com/challenges/flextrack-challenge-2025/dataset_files](https://www.aicrowd.com/challenges/flextrack-challenge-2025/dataset_files).
4. Clone this repository by : 
```
git clone git@gitlab.aicrowd.com:aicrowd/challenges/flextrack-challenge-2025/flextrack-challenge-2025-starter-kit.git
cd flextrack-challenge-2025-starter-kit
```
5. Move the dataset files into the `data/` directory.
6. Install requirements using:
```
pip install -r requirements.txt
```
7. **Run [`random_submission.py`](random_submission.py)** — a baseline script for generating a random submission.

6. Run `random_submission.py` locally. This will generate a sample submission file:
   ```
   my-submission-v0.1.csv
   ```
7. Upload the sample submission file to [the challenge page](https://www.aicrowd.com/challenges/flextrack-challenge-2025/) by clicking on the **Create Submission** button.
8. Check your scores on the [Leaderboard](https://www.aicrowd.com/challenges/flextrack-challenge-2025/leaderboards) of the competition.




**Note:** Additional baselines may be provided as the competition progresses. 

---

# 🧠 Tasks

The competition has two tasks:

## Task 1: Classification (Phase 1)

* Predict **demand response flags** (-1, 0, +1) from building time-series data.
* Submission: CSV with (Site, Timestamp, Predicted Flag).
* Metric: **Geometric Mean Score** (with F1-score also shown).

## Task 2: Regression (Phase 2)

* Predict **demand response capacity** (kW).
* Submission: CSV with (Site, Timestamp, Predicted Flag, Predicted Capacity).
* Metric: Details will be announced at Phase 2 launch.


---

# 📈 Evaluation

Please check the [local_evaluation.py](local_evaluation.py) for a walk through of how we will be calculating your scores for each of the tasks. 

---

# 🏆 Prizes

💰 **Total Prize Pool: \$20,000 AUD** (cash + travel grants).

| Prize     | Leaderboard | Travel Grant |
| --------- | ----------- | ------------ |
| 1st Place | \$5,000     | \$5,000      |
| 2nd Place | \$3,000     | \$3,000      |
| 3rd Place | \$1,000 × 2 | \$1,000 × 2  |

🏅 Travel Grants support in-person presentation at the **2025 IEEE International Conference on Energy Technologies for Future Grids**.

---

# 📘 Submission & Rules

* Up to **10 submissions per day**.
* Top teams must submit code via private GitHub/GitLab for validation.
* Only organiser-provided datasets are allowed.
* Winners must submit detailed solution documentation (template to be provided).

---

# 🏛️ Conference

Winners will be invited to present at:

📍 **2025 IEEE International Conference on Energy Technologies for Future Grids**

* Date: 7–11 December 2025
* Location: University of Wollongong, Australia
* [Conference Website](https://attend.ieee.org/etfg-2025)

Optional: Submit a paper (deadline **30 Sept 2025**) — [submission link](https://uow.au1.qualtrics.com/jfe/form/SV_8DkK1fkcSp6s3rg).

---

# 🤝 Acknowledgements

FlexTrack is part of the **NSW Digital Infrastructure for Energy Flexibility** project, funded by the NSW Government with CSIRO, under the **Net Zero Plan Stage 1: 2020–2030**, and supported by **RACE for 2030 Cooperative Research Centre**.
