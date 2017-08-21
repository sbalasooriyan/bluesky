# -*- coding: utf-8 -*-
"""
Created on Sun Apr 02 21:18:08 2017

@author: suthe
"""
import re

def cmd_eval(selfobj, cmd_text):
    # Useful for interacting in Python shell while in BlueSky
    # Use '?' before letters that must be capitalized
    # Can also enclose with '??'

    # Make command lowercase
    cmd_text = cmd_text.lower()
    # Replace 'self' with 'selfobj'
    cmd_text = cmd_text.replace('self','selfobj')
    # Capitalize letter enclosed by '??'
    cmd_text = re.sub(r'\?\?(.+?)\?\?', lambda m: m.group(1).upper(), cmd_text)
    # Capitalize letters after '?'
    cmd_text = re.sub(r'\?(\w)', lambda m: m.group(1).upper(), cmd_text)
    # Replace 'selfobj' with 'self'
    cmd_text = cmd_text.replace('selfobj','self')
    # Current output
    output = 'Issued command-line:\n>> ' + cmd_text
    # Return
    return output, cmd_text
    
def cmd_eval_correct(selfobj, cmd_text, output):
    # It has limited capabilities to correct case-sensitive mistakes
    # By looking through builtin-functions, objects and functions within objects

    # Replace SELF with SELFOBJ
    cmd_text = cmd_text.replace('self','selfobj')
    # Find opening and closing parenthesis
    paren_1 = [i for i, lt in enumerate(cmd_text) if lt == "("]
    paren_2 = [i for i, lt in enumerate(cmd_text) if lt == ")"]
    periods = [i for i, lt in enumerate(cmd_text) if lt == "."]
    
    # Lengths
    N_paren  = len(paren_1)
    N_period = len(periods)
    # Position of first period
    if N_period == 0:
        first_period = len(cmd_text)
    else:    
        first_period = periods[0]
    # Check builtins, iteratively find them
    # It is expected that builtins are found before '(', this bool keeps track
    builtins_bool = [False] * N_paren
    # All builtin functions of Python
    builtins_item = dir(__builtins__)
    # Loop through parenthesis, check if a builtin function is used
    for i in range(N_paren):
        # Find starting index (either 0 or previous '(')
        if i == 0:
            start = 0
        else:
            start = paren_1[i-1] + 1
        # It is assumed that builtins are used AROUND any objects
        if start < first_period:
            # The partial command that might be a builtin-function
            cmd_part = cmd_text[start:paren_1[i]]
            # Check against builtins (case-insensitive)
            builtins_index = [j for j, items in enumerate(builtins_item) if cmd_part.lower() == items.lower()]
            # If found to be builtin...
            if len(builtins_index) == 1:
                # Set bool to True
                builtins_bool[i] = True
                # Check if cases are correct, if not replace with builtin...
                if cmd_part != builtins_item[builtins_index[0]]:
                    cmd_text = cmd_text[0:start] + builtins_item[builtins_index[0]] + cmd_text[paren_1[i]:]
    
    # This is the assumed form, where the amount varies but order does not...
    # builtin(builtin(object.object.var(args)))
    # For robustness, check that not only builtins were given (which is possible!)
    if sum(builtins_bool) < N_paren:
        # Indices of False builtins
        ind = [i for i, elements in enumerate(builtins_bool) if not elements]
        # Get slice start/stop
        if ind[0] == 0:
            start = 0
        else:
            start = paren_1[ind[0]-1] + 1
        if ind[0] == N_paren - 1:
            stop = paren_2[0]
        else:
            stop = paren_1[ind[0]]
        # It is assumed that these contain actual obj/vars
        # Bool to check whether cmd_text has changed
        change = False
        # Loop through period
        for i in range(N_period+1):
            # Get indices of objects in start2 and stop2
            if i == 0:
                start2 = start
            else:
                start2 = periods[i-1] + 1
            if start2 < stop:
                # cmd_text should be replaced
                change = True
                if i == N_period:
                    stop2 = stop
                elif periods[i] > stop:
                    stop2 = stop
                else:
                    stop2 = periods[i]
                # For the first object check if in globals
                # The second in dir, since it also lists functions
                if i == 0:
                    itemlist = globals().keys()
                else:
                    itemlist = dir(eval(cur))
                # The object
                cmd_part = cmd_text[start2:stop2]
                locals_index = [j for j, item in enumerate(itemlist) if cmd_part.lower() == item.lower()]
                # If something is found, replace cmd_part
                if len(locals_index) > 0:
                    cmd_part = itemlist[locals_index[0]]
                # Add cmd_part to current object.object.object...
                if i == 0:
                    cur = cmd_part
                else:
                    cur += "." + cmd_part
        # If cmd_text has changed, replace with cur
        if change:
            cmd_text = cmd_text[0:start] + cur + cmd_text[stop:]
    cmd_text = cmd_text.replace('selfobj','self')
    output += '\n' + 'Corrected command-line:\n>> ' + cmd_text
    # Return
    return output, cmd_text

def clipboard(oper="get", data=""):
    # From https://stackoverflow.com/a/3429034

    # Imports
    import ctypes
    
    # Open Cliboard
    ctypes.windll.user32.OpenClipboard(None)
    
    # Check if we get or set Clipboard
    if oper == "get":
        # Get clipboard
        pcontents = ctypes.windll.user32.GetClipboardData(1)
        # Decode
        data = ctypes.c_char_p(pcontents).value
        # Close clipboard
        ctypes.windll.user32.CloseClipboard
        # Return data
        return data
    elif oper == "set":
        # Empty Clipboard
        ctypes.windll.user32.EmptyClipboard()
        # Allocate memory
        hcd = ctypes.windll.kernel32.GlobalAlloc(0x2000, len(bytes(data)) + 1)
        # Lock memory
        pchData = ctypes.windll.kernel32.GlobalLock(hcd)
        # Encode and copy
        ctypes.cdll.msvcrt.strcpy(ctypes.c_char_p(pchData), bytes(data))
        # Free memory
        ctypes.windll.kernel32.GlobalUnlock(hcd)
        # Set Clipboard
        ctypes.windll.user32.SetClipboardData(1, hcd)
        # Close clipboard
        ctypes.windll.user32.CloseClipboard()
        # Return Succes
        return True
    else:
        print "Invalid arg in clipboard"
        # Return Fail
        return False