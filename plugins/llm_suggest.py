#TODO write a description for this script
#@author 
#@category Python 3
#@keybinding Ctrl-Shift-K
#@menupath 
#@toolbar 


#TODO Add User Code Here
from ghidra.util.task import TaskMonitor
from ghidra.app.decompiler import DecompInterface
from ghidra.app.decompiler import DecompileOptions
from ghidra.program.model.symbol import SourceType
from ghidra.util.task import ConsoleTaskMonitor
import json
from java.security import MessageDigest
from ghidra.program.model.listing import Program
import requests


def send_code_to_server(sha256, function_offset, code, suggestion):
    """
    Send the decompiled code to a server for analysis.
    
    Args:
        sha256 (str): The SHA-256 hash value of the program.
        function_offset (str): The offset value of the function.
        code (str): The decompiled code of the function.
    
    Returns:
        dict: A dictionary containing the server's response.
    """
    url = "http://localhost:8000/suggest"  # Adjust the URL if needed


    print(sha256, function_offset, code, suggestion)
    data = {
        "sha256": sha256,
        "offset": function_offset,
        "code": code,
        "suggestion": suggestion
    }
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, data=json.dumps(data), headers=headers)
    print(response)
    
    if response.status_code != 200:
        raise Exception(f"Server returned status code {response.status_code}: {response.text}")
    
    return response.json()



options = DecompileOptions()
monitor = ConsoleTaskMonitor()
ifc = DecompInterface()
ifc.setOptions(options)

state = getState()
location_str = state.getCurrentLocation().getAddress().toString()
location = state.getCurrentLocation().getAddress()

print("Starting script")
task_monitor = TaskMonitor.DUMMY
print("Task monitor created")
decompiler = DecompInterface()
print("Decompiler created")
decompiler.openProgram(currentProgram())
print("Program opened at: {location}".format(location=location))


hash_value = currentProgram().getExecutableSHA256()
fm = currentProgram().getFunctionManager()
#first we try to get the function at the current location

function = fm.getFunctionAt(location)
print("Function found at location: {location}".format(location=location))
#check to see if Fucntion is None
if function is None:
    print("Function is None, looking for a function with that address")
    function = fm.getFunctionContaining(location)
    #check to see if Fucntion is None
    if function is None:
        print("Function is None for this address")

    

#hash_value = program.getExecutableImage().getBytes()

#print("SHA-256 Hash: {hash_value}".format(hash_value=hash_value))
#print("location_str: {location_str}".format(location_str=location_str))
#print("function: {function}".format(function=function))

c_code = decompiler.decompileFunction(function, 90, task_monitor).getDecompiledFunction().getC()
#print(c_code)
comment = function.getComment()
#print("repeatable comment: {comment}".format(comment=comment))

response = send_code_to_server(hash_value, location_str, c_code, comment)
print("Server Response:", response)
