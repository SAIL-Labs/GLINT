import numpy as np

def strehl_to_wfe(strehl, l = 1.55):
    return l/(2*np.pi)*np.sqrt(-1*np.log(strehl))


while True:
    user_input = input("Enter Strehl: ")
    
    if user_input.lower() == "done":
        break

    try:
        strehl = float(user_input)
        wfe = strehl_to_wfe(strehl)
        print("WFE: {:.4f}".format(wfe))

    except ValueError:
        print("Please enter a valid number.")
