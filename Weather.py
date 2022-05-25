{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled17.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyMJ6qunH+aa5hbcsdSIYhsl",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/NagulaThoshithanjali/weather/blob/main/Untitled17.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "VbwGQZOgAeFS"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.ensemble import RandomForestRegressor\n",
        "creation=pd.read_csv('temps2.csv', sep=';')\n",
        "creation.head(5)\n",
        "print(creation)\n",
        "print(\"The shape of our feature is:\", creation.shape)\n",
        "creation.describe()\n",
        "creation=pd.get_dummies(creation)\n",
        "creation.iloc[:,5:].head(5)\n",
        "labels=np.array(creation['Temperature'])\n",
        "creation=creation.drop('Temperature',axis=1)\n",
        "creation_list=list(creation.columns)\n",
        "creation=np.array(creation)\n",
        "train_creation, test_creation, train_labels, test_labels= train_test_split(creation,labels, test_size=0.30,random_state=4)\n",
        "print('Training creation shape:', train_creation.shape)\n",
        "print('Training labels shape:', train_labels.shape)\n",
        "print('Testing creation shape:', test_creation.shape)\n",
        "print('Testing label shape:', test_labels.shape)\n",
        "rf=RandomForestRegressor(n_estimators=1000, random_state=4)\n",
        "rf.fit(train_creation, train_labels)\n",
        "predictions=rf.predict(test_creation)\n",
        "print(predictions)\n",
        "errors=abs(predictions - test_labels)\n",
        "print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')\n",
        "mape=100* (errors/test_labels)\n",
        "accuracy=100-np.mean(mape/3)\n",
        "print('Accuracy of the model:', round(accuracy,2),'%') "
      ]
    }
  ]
}
