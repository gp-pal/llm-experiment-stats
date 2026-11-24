## Why A/B Testing?

A/B testing (also known as split testing) is a powerful statistical method for
comparing two or more versions of a product or service to determine which performs
better. By randomly assigning users to different groups, we can control for 
confounding variables and isolate the **treatment effect**.

### Application in Marketing

In marketing, A/B testing is widely used to refine and improve key elements such as:

 - Increasing email open and click-through rates
 - Optimizing social media ad performance
 - Enhancing product recommendations on websites 
 - And many other use cases

## Datasets Overview

### A/B Testing Projects: Fast Food Marketing & Cookie Cats

This project showcases A/B testing analysis on two distinct datasets from different domains:
a **fast food marketing campaign** and a **mobile game player experiment**. 
Both experiments aim to test whether specific interventions (marketing strategies or game
design changes) significantly affect user behavior.

### 1. Fast Food Marketing Campaign A/B Test

This dataset comes from a simulated fast food marketing experiment that tests the 
effectiveness of **three different promotional plans**.

### 2. Cookie Cats A/B Test

This dataset is derived from the mobile game **Cookie Cats**, where developers 
tested a change in game design: the placement of a **gate** that blocks player 
progress until a task is completed.

## Setting Up the Local Environment

To avoid installing the Python interpreter globally on your computer,
you can use a virtual environment. Follow these steps to create and
use a virtual environment:

### 1. Install `virtualenv`

If you don't have `virtualenv` installed, you can install it using pip:

```bash
pip install virtualenv
```

### 2. Create a Virtual Environment

Navigate to your project directory and create a virtual environment:

```bash
virtualenv venv
```

### 3. Activate the Virtual Environment

- On Linux or macOS:
  ```bash
  source venv/bin/activate
  ```
- On Windows:
  ```bash
  .\venv\Scripts\activate
  ```
  
### 4. Install the Required Packages

With the virtual environment activated, install the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

This way, you can use a local environment without installing the Python interpreter globally on your computer.

### Jupyter Notebook Installation

- To install Jupyter Notebook, run the following command:

```sh
  pip install notebook
```

### Using Jupyter Notebook

To use Jupyter Notebook, follow these steps:

- Open Jupyter Notebook by running:

```sh
  jupyter notebook
```
   
## Setup and running Streamlit

1. Install Streamlit:

```bash
  pip install streamlit
```

2. Run the Streamlit app:

```bash
  streamlit run dashboard.py
```