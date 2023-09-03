#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.stats import norm
import numpy as np
import scipy as sq
import time
import pandas as pd
import sys
from scipy.special import log_ndtr, ndtr
import math


# In[ ]:


c = 0

for k in range(1,6):

    df = pd.read_excel(r'C:\Users\Akash Prasad\Live1.xlsx', sheet_name='Sheet2')

    start_time = time.time()

    # Functions of the script

    def Call_BS_Value(S, X, r, T, v):

        # Calculates the value of a call option (Just the Black-Scholes formula for call options)

        # S is the share price at time T

        # X is the strike price

        # r is the risk-free interest rate

        # T is the time to maturity in years (days/365)

        # v is the volatility

       

        d_1 = (np.log(S / X) + (r + v ** 2 * 0.5) * T) / (v * np.sqrt(T))

        d_2 = d_1 - v * np.sqrt(T)

        return S * ndtr(d_1) - X * np.exp(-r * T) * ndtr(d_2) #returns the value of the call option price predicted by the model

 

    ''''

    def Call_IV_Obj_Function(S, X, r, T, v, Call_Price):

        # Objective function which sets market and model prices equal to zero (Function needed for Call_IV)

        # The parameters are explained in the Call_BS_Value function

        return Call_Price - Call_BS_Value(S, X, r, T, v)

 

    def Call_IV(S, X, r, T, Call_Price, a=-2, b=2, xtol=0.000001):

        # Calculates the implied volatility for a call option with Brent's method

        # The first four parameters are explained in the Call_BS_Value function

        # Call_Price is the price of the call option

        # Last three variables are needed for Brent's method

 

        def fcn(v):

            return Call_IV_Obj_Function(S, X, r, T, v, Call_Price)

 

        try:

            result = sq.optimize.brentq(fcn, a=a, b=b, xtol=xtol)

            return np.nan if result <= xtol else result

        except ValueError:

            return np.nan

 

    def Put_BS_Value(S, X, r, T, v):

        # Calculates the value of a put option(Just the Black-Scholes formula for put options)

        # The parameters are explained in the Call_BS_Value function

        d_1 = (np.log(S / X) + (r + v ** 2 * 0.5) * T) / (v * np.sqrt(T))

        d_2 = d_1 - v * np.sqrt(T)

        return X * np.exp(-r * T) * ndtr(-d_2) - S * ndtr(-d_1)

 

    def Put_IV_Obj_Function(S, X, r, T, v, Put_Price):

        # Objective function which sets market and model prices equal to zero (Function needed for Put_IV)

        # The parameters are explained in the Call_BS_Value function

        return Put_Price - Put_BS_Value(S, X, r, T, v)

 

    '''

    '''

    def Put_IV(S, X, r, T, Put_Price, a=-0.01, b=0.01, xtol=0.000001):

        # Calculates the implied volatility for a put option with Brent's method

        # The first four parameters are explained in the Call_BS_Value function

        # Put_Price is the price of the call option

        # Last three variables are needed for Brent's method

 

        def fcn(v):

            return Put_IV_Obj_Function(S, X, r, T, v, Put_Price)

 

        try:

            result = sq.optimize.brentq(fcn, a=a, b=b, xtol=xtol)

            return np.nan if result <= xtol else result

        except ValueError:

            return np.nan

    '''

 

    def Calculate_IV_Call_Put(S, X, r, T, Option_Price, Put_or_Call):

        # This is a general function witch summarizes Call_IV and Put_IV (delivers the same results)

        # Can be used for a Lambda function within Pandas

        # The first four parameters are explained in the Call_BS_Value function

        # Put_or_Call:

        # 'C' returns the implied volatility of a call

        # 'P' returns the implied volatility of a put

        # Option_Price is the price of the option.

       

        def call_price_diff(v):

            d_1 = (np.log(S / X) + (r + v ** 2 * 0.5) * T) / (v * np.sqrt(T))

            d_2 = d_1 - v * np.sqrt(T)

            bs_call_price =  S * ndtr(d_1) - X * np.exp(-r * T) * ndtr(d_2)

            return Option_Price - bs_call_price

 

       

        def put_price_diff(v):

                d_1 = (np.log(S / X) + (r + v ** 2 * 0.5) * T) / (v * np.sqrt(T))

                d_2 = d_1 - v * np.sqrt(T)

                bs_put_price = X * np.exp(-r * T) * ndtr(-d_2) - S * ndtr(-d_1)

                return Option_Price - bs_put_price

 

        xtol=0.000001

        if Put_or_Call == 'C':

            try:

                result = sq.optimize.brentq(call_price_diff, a=-0.01, b=0.01, xtol = xtol)

                return np.nan if result <= xtol else result

            except ValueError:

                return np.nan

        if Put_or_Call == 'P':

            try:

                result = sq.optimize.brentq(put_price_diff, a=-0.01, b=0.01, xtol = xtol)

                return np.nan if result <= xtol else result

            except ValueError:

                return np.nan

            #return Put_IV(S, X, r, T, Option_Price)

        else:

            return 'Neither call or put'

 

    #Body of the script

    S = []

    S = df['Fwd']

    #for item in S:

        #float(item)

    #print(type(S[i]))

    X = []

    X = df['Strike']

    #for item in X:

        #float(item)

    r = []

    r =  np.empty(S.size, dtype = float)

    T = []

    T = df['DaystoExpiry']

    #for item in T:

        #float(item)

    Option_Price = []

    Option_Price = df['BANKNIFTY']

    #for item in Option_Price:

        #float(item)

    Type = []

    Type = df['Type']

   

    #Call_Price = []

    #Call_Price = Call_IV(S,X,r,T,Option_Price)

    Call_Summarize = np.empty(S.size, dtype = float)

    for i in range(0,S.size):

        Call_Summarize[i] = (Calculate_IV_Call_Put(S[i],X[i],r[i],T[i],Option_Price[i],Type[i]))

    #Put_Price = []

    #Put_Price=Put_IV(S,X,r,T,Option_Price)

    #Put_Summarize = []

    #Put_Summarize=Calculate_IV_Call_Put(S,X,r,T,Option_Price,Type)

    #The output variables

    #print(type(r))

 

 
    '''

    def Calculate_IV_Call_Put(S, X, r, T, Option_Price, Put_or_Call):

        # This is a general function witch summarizes Call_IV and Put_IV (delivers the same results)

        # Can be used for a Lambda function within Pandas

        # The first four parameters are explained in the Call_BS_Value function

        # Put_or_Call:

        # 'C' returns the implied volatility of a call

        # 'P' returns the implied volatility of a put

        # Option_Price is the price of the option.

 

        if Put_or_Call == 'C':

            return Call_IV(S, X, r, T, Option_Price)

        if Put_or_Call == 'P':

            return Put_IV(S, X, r, T, Option_Price)

        else:

            return 'Neither call or put'

