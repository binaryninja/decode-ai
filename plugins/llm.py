#TODO write a description for this script
#@author 
#@category Python 3
#@keybinding Ctrl-Shift-L
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


def send_code_to_server(sha256, function_offset, code):
    """
    Send the decompiled code to a server for analysis.
    
    Args:
        sha256 (str): The SHA-256 hash value of the program.
        function_offset (str): The offset value of the function.
        code (str): The decompiled code of the function.
    
    Returns:
        dict: A dictionary containing the server's response.
    """
    url = "http://localhost:8000/evaluate_local"  # Adjust the URL if needed
    data = {
        "sha256": sha256,
        "function_offset": function_offset,
        "code": code
    }
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, data=json.dumps(data), headers=headers)
    
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

print("SHA-256 Hash: {hash_value}".format(hash_value=hash_value))
print("location_str: {location_str}".format(location_str=location_str))
print("function: {function}".format(function=function))

c_code = decompiler.decompileFunction(function, 90, task_monitor).getDecompiledFunction().getC()
print(c_code)
rcomment = function.getRepeatableComment()
print("repeatable comment: {rcomment}".format(rcomment=rcomment))

response = send_code_to_server(hash_value, location_str, c_code)
print("Server Response:", response)

#Server Response: {'llm_name': 'int_array_calculation', 'llm_short_summary': 'This function takes three parameters: two arrays of integers and a pointer to an integer. It appears to perform some sort of calculation on the two input arrays and stores the results in the output array. The function uses a lot of local variables, which suggests that it may be trying to obfuscate its true purpose.', 'llm_step_by_step_description': '[INST]<<SYS>> You are an advanced malware reverse engineer capable of understanding decompiled C code and identifying malicious functionality<</SYS>>\n\nYou must output a descriptive ### Function Summary that describes the following decompiled code followed by a descriptive ### New Function Name \n\nDo not provide any extra information in the ### New Function Name. Only provide the name of the function. Do not include any extra information.[/INST]\n\n### Code:\n\nvoid __fastcall FUN_00406150(int *param_1,int *param_2,int *param_3)\n\n{\n  int iVar1;\n  int iVar2;\n  int iVar3;\n  int iVar4;\n  int iVar5;\n  int iVar6;\n  int iVar7;\n  int iVar8;\n  int iVar9;\n  int iVar10;\n  int iVar11;\n  int iVar12;\n  int iVar13;\n  int iVar14;\n  int iVar15;\n  int iVar16;\n  int iVar17;\n  int iVar18;\n  \n  iVar1 = param_2[1];\n  iVar2 = param_2[5];\n  iVar3 = param_3[1];\n  iVar4 = param_3[5];\n  iVar5 = param_2[7];\n  iVar6 = param_3[7];\n  iVar7 = param_2[2];\n  iVar8 = param_3[2];\n  iVar9 = param_2[3];\n  iVar10 = param_3[3];\n  iVar11 = param_2[4];\n  iVar12 = param_3[4];\n  iVar13 = param_2[6];\n  iVar14 = param_3[6];\n  iVar15 = param_2[8];\n  iVar16 = param_3[8];\n  iVar17 = param_2[9];\n  iVar18 = param_3[9];\n  *param_1 = *param_2 + *param_3;\n  param_1[1] = iVar1 + iVar3;\n  param_1[2] = iVar7 + iVar8;\n  param_1[3] = iVar9 + iVar10;\n  param_1[4] = iVar11 + iVar12;\n  param_1[6] = iVar13 + iVar14;\n  param_1[7] = iVar5 + iVar6;\n  param_1[5] = iVar2 + iVar4;\n  param_1[8] = iVar15 + iVar16;\n  param_1[9] = iVar17 + iVar18;\n  return;\n}\n\n\n\n### Function Summary:\n\nThis function takes three parameters: two arrays of integers and a pointer to an integer. It appears to perform some sort of calculation on the two input arrays and stores the results in the output array. The function uses a lot of local variables, which suggests that it may be trying to obfuscate its true purpose.\n\n### New Function Name:\n\nint_array_calculation'}

#set function name to llm_name
function.setName(response["llm_name"], SourceType.DEFAULT)
#set function comment to llm_short_summary
function.setComment(response["llm_short_summary"])
#set function repeatable comment to llm_step_by_step_description
