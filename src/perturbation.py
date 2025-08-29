import javalang
import re
import string
import random
import csv
from collections import Counter
import traceback

#def operand_mutation(X_test, y_test):

def generate_random_variable_name(length=8):
    """Generate a random variable name of given length."""
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def is_kotlin_code(code_snippet):
    # List of keywords and features more common in Kotlin than Java
    kotlin_indicators = [
        "val ",  # Read-only local variables
        "var ",  # Mutable local variables
        "fun ",  # Function declaration
        "?:",    # Elvis operator
        "?.",    # Safe call operator
        "::",    # Callable reference
        "suspend ", # Coroutines
        "data class ", # Data classes
        "object ", # Object declaration
    ]
    
    # Check if any of the Kotlin indicators are present in the code snippet
    for indicator in kotlin_indicators:
        if indicator in code_snippet:
            return True
    
    return False


import javalang
import re
from collections import Counter

def renaming_top_k_variables(java_code, new_variable_names, k=3):
    """
    Renames the top k most used local variables (not class fields or fields accessed via 'this.') in the given Java code.
    Only variables declared in method bodies or as parameters are considered.
    """
    import javalang
    from collections import Counter
    import re

    code_wrapped = False
    original_code = java_code
    if not re.match(r'\s*(public\s+)?class\s+\w+', java_code):
        java_code = f'public class DummyClass {{\n{java_code}\n}}'
        code_wrapped = True

    try:
        tree = javalang.parse.parse(java_code)
        local_vars = set()
        usage_counter = Counter()
        param_vars = set()

        # Collect method parameter names
        for _, node in tree.filter(javalang.tree.MethodDeclaration):
            if node.parameters:
                for param in node.parameters:
                    if hasattr(param, 'name'):
                        param_vars.add(param.name)

        # Collect local variable names (including loop variables)
        for _, node in tree.filter(javalang.tree.LocalVariableDeclaration):
            for declarator in node.declarators:
                if hasattr(declarator, 'name'):
                    local_vars.add(declarator.name)
        for _, node in tree.filter(javalang.tree.ForStatement):
            if node.control and hasattr(node.control, 'init') and node.control.init:
                for init in node.control.init:
                    if hasattr(init, 'declarators'):
                        for declarator in init.declarators:
                            if hasattr(declarator, 'name'):
                                local_vars.add(declarator.name)

        all_local_vars = local_vars | param_vars

        # Count usages of each local variable (excluding field accesses)
        for _, node in tree.filter(javalang.tree.MemberReference):
            if node.qualifier is None and node.member in all_local_vars:
                usage_counter[node.member] += 1
        for _, node in tree.filter(javalang.tree.VariableDeclarator):
            if node.name in all_local_vars:
                usage_counter[node.name] += 1
        for _, node in tree.filter(javalang.tree.FormalParameter):
            if node.name in all_local_vars:
                usage_counter[node.name] += 1

        if not usage_counter:
            return original_code  # No local variables to rename

        # Find the k most used local variables
        most_used_vars = [var for var, _ in usage_counter.most_common(k)]
        if not new_variable_names or not isinstance(new_variable_names, list):
            raise ValueError("new_variable_names must be a list of strings")
        # Pick k new names that do not conflict
        new_names = []
        used_names = set(all_local_vars)
        for candidate in new_variable_names:
            if candidate not in used_names:
                new_names.append(candidate)
            if len(new_names) == len(most_used_vars):
                break
        if len(new_names) < len(most_used_vars):
            return original_code  # Not enough new names

        # Replace only the local variables (not field accesses like this.sharedState)
        for old_var, new_var in zip(most_used_vars, new_names):
            pattern = rf'(?<![\w.]){re.escape(old_var)}(?![\w])'
            java_code = re.sub(pattern, new_var, java_code)

        # If we wrapped, extract the code back out
        if code_wrapped:
            match = re.search(r'public class DummyClass \{(.*)\}', java_code, re.DOTALL)
            if match:
                java_code = match.group(1).strip()
    except Exception as e:
        print(f"Error during renaming top k variables: {str(e)} ; {str(new_variable_names)}")
        return original_code

    return java_code

def renaming_least_used_variable_in_60_percent_methods(java_code, new_variable_names):
    """
    Renames the top 3 most used local variables in 1% of the methods (randomly selected, at least 1) in the given Java code.
    Only variables declared in method bodies or as parameters are considered.
    """
    import javalang
    import random
    import re
    # Defensive: filter new_variable_names to only valid strings
    if not new_variable_names or not isinstance(new_variable_names, list):
        return java_code
    new_variable_names = [n for n in new_variable_names if isinstance(n, str) and n is not None]
    if not new_variable_names:
        return java_code

    try:
        tree = javalang.parse.parse(java_code)
        # Collect all method declarations with their positions
        method_nodes = []
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            if hasattr(node, 'position') and node.position:
                method_nodes.append((node, node.position))

        if not method_nodes:
            return java_code  # No methods to process

        # Randomly select 1% of the methods (at least 1)
        num_to_select = max(1, int(len(method_nodes) * 1))
        selected_methods = set(random.sample(method_nodes, num_to_select))

        # Split code into lines for easier manipulation
        code_lines = java_code.splitlines()
        # To avoid overlapping edits, process from last to first
        method_nodes_sorted = sorted(method_nodes, key=lambda x: x[1].line, reverse=True)

        for node, pos in method_nodes_sorted:
            if (node, pos) not in selected_methods:
                continue  # Skip methods not selected
            # Find method start and end lines
            start_line = pos.line - 1  # 0-based
            # Try to find the end of the method by counting braces
            brace_count = 0
            method_start = None
            method_end = None
            for i in range(start_line, len(code_lines)):
                line = code_lines[i]
                if '{' in line:
                    if method_start is None:
                        method_start = i
                    brace_count += line.count('{')
                if '}' in line:
                    brace_count -= line.count('}')
                if method_start is not None and brace_count == 0:
                    method_end = i
                    break
            if method_start is None or method_end is None:
                continue  # Could not find method body
            # Extract method code
            method_code = '\n'.join(code_lines[method_start:method_end+1])
            if not method_code or not isinstance(method_code, str):
                continue  # Defensive: skip if method_code is None or empty
            # Wrap method code in a dummy class for parsing
            wrapped_method_code = f'public class DummyClass {{\n{method_code}\n}}'
            try:
                # Rename top 3 most used variables in this method only
                changed_method_code = renaming_top_k_variables(wrapped_method_code, new_variable_names, k=3)
                # Extract code back out of dummy class
                match = re.search(r'public class DummyClass \{(.*)\}', changed_method_code, re.DOTALL)
                if match:
                    changed_method_code = match.group(1).strip()
            except Exception as e:
                print(f"Skipping method due to parse error: {str(e)}\nProblematic method code:\n{method_code}")
                continue
            # Replace in code_lines
            code_lines = code_lines[:method_start] + changed_method_code.splitlines() + code_lines[method_end+1:]
        # Reconstruct code
        return '\n'.join(code_lines)
    except Exception as e:
        print(f"Error during 1% top-k method renaming: {str(e)} ; {str(new_variable_names)}")
        return java_code

def variableRenaming_perturbation(java_code, idx, new_variable_names, perturb_cat):
    count_kotlin = 0
    java_code_wrapped = f"""
public class WrapperClass {{
{java_code}
}}
    """
    if not is_kotlin_code(java_code):
        changed_code = renaming_least_used_variable_in_60_percent_methods(java_code_wrapped, new_variable_names)
        # Remove the wrapper class and preserve original indentation
        pattern = r'public class WrapperClass \{\n(.*)\n\}'
        match = re.search(pattern, changed_code, re.DOTALL)
        if match:
            changed_code = match.group(1).rstrip("\n")
    else:
        changed_code = java_code  # not modifying kotlin code for now
        count_kotlin += 1
    return changed_code


def multiLine_perturbation_with_async(feature_types="Most_Imp"):
    comment_start = "/**\n"
    if feature_types == "Most_Imp":
    #For most important features
        sleep_statement = "* Thread sleep is inappropriately used in this test. In the current settings, 1000 ms is not enough to wait. Hence, increasing the sleeping time for thread to 5000 ms, ultimately assertion value of this test also changed.\n"
    else:
        #For least important features
        sleep_statement = "* tfs Shuffle Uninterruptibles. Settings some random address made crash the savings of the metadata. Auto session is Optional. Insert multiCalculator and enable agent's build. The lifecycle can pause. Any interruption will get IOException. \n"
    comment_end = "**/\n"
    # Combine the while loop, sleep statements, and closing brace
    code_to_add = comment_start + sleep_statement + comment_end
    return code_to_add


def multiLine_perturbation_with_conc(feature_types="Most_Imp"):
    comment_start = "/**\n"
    if feature_types == "Most_Imp":
        #For most important features
        sleep_statement = "* AtomicInteger is robust. Here, the job type was missing for this thread. Thread will call the sleep with 10 seconds. We are making concurrent program. CountdownLatch is also used here. \n"
    else:
        #For least important features
        sleep_statement = "* The current value of the Counter is 100. Solver has completed the task successfully. The variable 'assign' has been initialized. Temporary storage 'Temp' is being used for processing. The Writable data structure is now available for use. Configuration 'conf' has been loaded successfully. \n"
    comment_end = "**/\n"
    # Combine the while loop, sleep statements, and closing brace
    code_to_add = comment_start + sleep_statement + comment_end
    return code_to_add

def multiLine_perturbation_with_time(feature_types="Most_Imp"):
    comment_start = "/**\n"
    if feature_types == "Most_Imp":
        #For most important features
        sleep_statement = "* Timer repetition. Date : date.now() Time: 10:01:05. Assert duration will be 1000 Seconds. dayOfweek=saturday. \n"
    else:
        #For least important features
        sleep_statement = "* Build started using Maven version. The RecyclerView is initializing with the correct indices. Network check: connection successful. Nano precision required for execution timing. Build failed due to network error.\n"
    comment_end = "**/\n"
    # Combine the while loop, sleep statements, and closing brace
    code_to_add = comment_start + sleep_statement + comment_end
    return code_to_add

def multiLine_perturbation_with_uc(feature_types="Most_Imp"):
    comment_start = "/**\n"
    if feature_types == "Most_Imp":
        #For most important features
        sleep_statement = "* JSON is added. previously this test had arrayList, and hash. This list of collection are now changed in the test with JSON. The assertion takes some samples and mock object is reset after serialization.\n"
    else:
        #For least important features
        sleep_statement = "* Each test case generates unique tokens. Txt file loaded successfully from the specified URI. Multimap inverse operation executed. Testing the Doc scheme for the current context. Assertion failed: Messages did not match.\n"
    comment_end = "**/\n"
    # Combine the while loop, sleep statements, and closing brace
    code_to_add = comment_start + sleep_statement + comment_end
    return code_to_add

def multiLine_perturbation_with_od(feature_types="Most_Imp"):
    comment_start = "/**\n"
    if feature_types == "Most_Imp":
        #For most important features
        sleep_statement = "* Directory Name was missing. File path is changed. Current file path will be the one directory back. The response for the method is handled based on the context.\n"
    else:
        #For least important features
        sleep_statement = "* Same code can be found in different countries. Country data has been updated in the repository. Find the records related to Germany in the database. The producer sent data to the destination (dst) successfully.\n"
    comment_end = "**/\n"
    # Combine the while loop, sleep statements, and closing brace
    code_to_add = comment_start + sleep_statement + comment_end
    return code_to_add

def multiLine_perturbation_with_nonflaky(feature_types="Most_Imp"):
    comment_start = "/**\n"
    if feature_types == "Most_Imp":
        #For most important features
        sleep_statement = "* The iteration throws Exception. assert condition is successful only when the string value is Primitive. In this test, the new object is put into a default object. Otherwise, it could throws exception.\n"
    else:
        sleep_statement = "* The same geometry across different APIs should shorten the multiple lookups. The specific windows have new views than the Linux. \n"
    comment_end = "**/\n"
    # Combine the while loop, sleep statements, and closing brace
    code_to_add = comment_start + sleep_statement + comment_end
    return code_to_add

def singleLine_perturbation_with_async(feature_types="Most_Imp"):
    if feature_types == "Most_Imp":
        #For most important features
        statement1 = "//System.out.println(\"Thread sleep is inappropriately used in this test. In the current settings, 1000 ms is not enough to wait.\");\n"
        statement2 = "//System.out.println(\"Setting sleeping time for thread to 5000 ms, ultimately assertion value of this test also changed.\");\n"
    else:
        #For least important features
        statement1 = "//System.out.println(\"tfs Shuffle Uninterruptibles. Settings some random address made crash the savings of the metadata. Auto session is Optional.\");\n"
        statement2 = "//System.out.println(\"Insert multiCalculator and enable agent's build.\");\n"
    # Combine the while loop, sleep statements, and closing brace
    code_to_add = statement1 + statement2
    return code_to_add

def singleLine_perturbation_with_conc(feature_types="Most_Imp"):
    if feature_types == "Most_Imp":
        #For most important features
        statement1 = "//System.out.println(\"AtomicInteger is robust. Here, the job type was missing for this thread.\");\n"
        statement2 = "//System.out.println(\"Thread will call the sleep with 10 seconds. We are making concurrent program. CountdownLatch is also used here.\");\n"
    else:
        #For least important features
        statement1 = "//System.out.println(\"The current value of the Counter is 100. Solver has completed the task successfully.\");\n"
        statement2 = "//System.out.println(\"The variable 'assign' has been initialized. The Writable data structure is now available for use. Configuration 'conf' has been loaded successfully.\");\n"
    # Combine the while loop, sleep statements, and closing brace
    code_to_add = statement1 + statement2
    return code_to_add

def singleLine_perturbation_with_time(feature_types="Most_Imp"):
    if feature_types == "Most_Imp":
        #For most important features
        statement1 = "//System.out.println(\"Timer repetition. Date : date.now() Time: 10:01:05. Assert duration will be 1000 Seconds. dayOfweek=saturday.\") \n"
        statement2 = "//System.out.println(\"Assert duration will be 1000 Seconds. dayOfweek=saturday.\") \n"
    else:
        #For least important features
        statement1 = "//System.out.println(\"Build started using Maven version. The RecyclerView is initializing with the correct indices.\");\n"
        statement2 = "//System.out.println(\"Network check: connection successful. Nano precision required for execution timing. Build failed due to network error.\");\n"

    code_to_add = statement1 + statement2
    return code_to_add

def singleLine_perturbation_with_uc(feature_types="Most_Imp"):
    if feature_types == "Most_Imp":
        #For most important features
        statement1 = "//System.out.println(\"The assertion takes some samples and mock object is reset after serialization.\");\n"
        statement2 = "//System.out.println(\"JSON is added. previously this test had arrayList, and hash. This list of collection are now changed in the test with JSON.\");\n"
    else:
        #For least important features
        statement1 = "//System.out.println(\"Each test case generates unique tokens. Txt file loaded successfully from the specified URI. Multimap inverse operation executed.\");\n"
        statement2 = "//System.out.println(\"Testing the Doc scheme for the current context. Assertion failed: Messages did not match.\");\n"
    # Combine the while loop, sleep statements, and closing brace
    code_to_add = statement1 + statement2
    return code_to_add

def singleLine_perturbation_with_od(feature_types="Most_Imp"):
    if feature_types == "Most_Imp":
        #For most important features
        statement1 = "//System.out.println(\"Directory Name was missing. File path is changed. Current file path will be the one directory back.\");\n"
        statement2 = "//System.out.println(\"The response for the method is handled based on the context.\");\n"
    else:
        #For least important features
        statement1 = "//System.out.println(\"Same code can be found in different countries. Country data has been updated in the repository.\");\n"
        statement2 = "//System.out.println(\"Find the records related to Germany in the database. The producer sent data to the destination (dst) successfully\");\n"

    code_to_add = statement1 + statement2
    return code_to_add

def singleLine_perturbation_with_nonflaky(feature_types="Most_Imp"):
    if feature_types == "Most_Imp":
        #For most important features
        statement1 = "//System.out.println(\"The iteration throws Exception. assert condition is successful only when the string value is Primitive.\");\n"
        statement2 = "//System.out.println(\"In this test, the new object is put into a default object. Otherwise, it could throws exception.\");\n"
    else:
        #For least important features
        statement1 = "//System.out.println(\"The same geometry across different APIs should shorten the multiple lookups.\");\n"
        statement2 = "//System.out.println(\"The specific windows have new views than the Linux.\");\n"

    # Combine the while loop, sleep statements, and closing brace
    code_to_add = statement1 + statement2
    return code_to_add

#def printStatement_perturbation_with_async():
#    #printStatement = "System.out.println(\"In the future, I will add Thread.sleep(1000) here. If we can wait a bit in that case thread syncronization might work better\");"
#
#    #For least important features
#    printStatement = "//System.out.println(\"This is renewed. The repository is now cloned. Failover the dataset is helpful.\");\n System.out.println(\"Failover the dataset is helpful.\");\n"
#    code_to_add = printStatement
#    return code_to_add
#
#def printStatement_perturbation_with_conc():
#    printStatement = "System.out.println(\"The type of job is AtomicInteger, which is robust enough.\");\nSystem.out.println(\"Latency is 5000ms\");"
#    code_to_add = printStatement
#    return code_to_add


def printStatement_perturbation_with_async(feature_types="Most_Imp"):
    if feature_types == "Most_Imp":
        #For most important features
        printStatement = "System.out.println(\"In the future, I will add Thread.sleep(1000) here. If we can wait a bit in that case thread syncronization might work better\");\n"
    else:
        #For least important features
        printStatement = "System.out.println(\"tfs Shuffle Uninterruptibles. Settings some random address made crash the savings of the metadata. Auto session is Optional. Insert multiCalculator and enable agent's build.\");\n"
    code_to_add = printStatement
    return code_to_add

def printStatement_perturbation_with_conc(feature_types="Most_Imp"):
    if feature_types == "Most_Imp":
        #For most important features
        printStatement = "System.out.println(\"The type of job is AtomicInteger, which is robust enough.\");\nSystem.out.println(\"Latency is 5000ms\");"
    else:
        #For least important features
        printStatement = "System.out.println(\"The current value of the Counter is 100. Solver has completed the task successfully.\");\nSystem.out.println(\"The variable assign has been initialized. The Writable data structure is now available for use.\");\n"
    code_to_add = printStatement
    return code_to_add

def printStatement_perturbation_with_time(feature_types="Most_Imp"):
    if feature_types == "Most_Imp":
        #For most important features
        printStatement = "System.out.println(\"The total duration is five hours\");\nSystem.out.println(\"The current date is date.now()\");"
    else:
        #For least important features
        printStatement = "System.out.println(\"Network check: connection successful. Nano precision required for execution timing. Build failed due to network error.\");\nSystem.out.println(\"Build started using Maven version\");\n"
    code_to_add = printStatement
    return code_to_add

def printStatement_perturbation_with_uc(feature_types="Most_Imp"):
    if feature_types == "Most_Imp":
        #For most important features
        printStatement = "System.out.println(\"This output of the List:\"+outputList);\nSystem.out.println(\"The sorted array is=\"+array);"
    else:
        #For least important features
        printStatement = "System.out.println(\"Txt file loaded successfully from the specified URI. Multimap inverse operation executed.\");\nSystem.out.println(\"Assertion failed: Messages did not match. URI scheme is valid: abc.\");\n"
    code_to_add = printStatement
    return code_to_add

def printStatement_perturbation_with_od(feature_types="Most_Imp"):
    if feature_types == "Most_Imp":
        #For most important features
        printStatement = "System.out.println(dirname);\n System.out.println(new PathData(dirString, path));"
    else:
        #For least important features
        printStatement = "System.out.println(\"Same code can be found in different countries. Country data has been updated in the repository.\");\nSystem.out.println(\"The producer sent data to the destination (dst) successfully.\");\n"
    code_to_add = printStatement
    return code_to_add

def printStatement_perturbation_with_nonflaky(feature_types="Most_Imp"): 
    if feature_types == "Most_Imp":
        #For most important features
        printStatement = "System.out.println(\"Property of the test\"+result.property());\n System.out.println(\"The assertion for this new test throws exception string.\");"
    else:
        #For least important features
        printStatement = "System.out.println(\"The same geometry across different APIs should shorten the multiple lookups.\");\nSystem.out.println(\"The specific windows have new views than the Linux.\");"
    code_to_add = printStatement
    return code_to_add

#********************************************************************DEADCODE********************************************
#************************************************************************************************************************
def deadcode_perturbation_with_conc(feature_types="Most_Imp"):
    while_loop = "while(false){\n    int i = 0;\n"
    if feature_types == "Most_Imp":
        #For most important features
        conc_statement = "    RobustJobScheduler robust = new RobustJobSchedulerWrapper(originalScheduler, l);\n    AtomicInteger count = new AtomicInteger();\n    JobHandler handle = robust.scheduleRecurring(""JobName"", 1, count::incrementAndGet);\n"
        #claude_suggestions = " try {\n Robust job = new Robust();\n job.schedule(new Query(), TEST);\n } catch (SQLState e) {   \n System.out.println("Unexpected SQLState: " + e.ttl);"
    else:
        #For least important features
        conc_statement = "    private DFSCluster dfsCluster;\n    private FSNamesystem fsNamesystem = new FSNamesystem();\n    LeaseManager leaseManager = new LeaseManager();\n    leaseManager.assignLease(\"user1\", dfsCluster);\n"
    closing_brace = "}\n"
    code_to_add = while_loop + conc_statement + closing_brace
    return code_to_add

def deadcode_perturbation_with_async(feature_types="Most_Imp"):
    while_loop = "while(false){\n    int i = 0;\n"
    #sleep_statement = "  Thread.sleep(1000);\n  i=i+5000;\n ConnectFuture connectFuture = vmPipeConnector.connect(vmPipeAddress);\n connectFuture.awaitUninterruptibly(); \n connectFuture.getSession().close(); \n"
    if feature_types == "Most_Imp":
        #For most important features
        claude_suggestions = "    try {\n   Thread.sleep(1000);  job.schedule(new Query(), TEST);\n    }catch (SQLState e) {\n        System.out.println(\"Unexpected SQLState: \" + e.ttl);\n    }\n"
    else:
        #For least important features
        claude_suggestions = "    if (tfs.address > 0) {\n        crashID = Random(0, 100);\n    }\n    text=""Auto session is optional"";\n    check(crashId, metadata);\n"
    closing_brace = "}\n"
    # Combine the while loop, sleep statements, and closing brace
    code_to_add = while_loop + claude_suggestions + closing_brace
    return code_to_add


def deadcode_perturbation_with_time(feature_types="Most_Imp"):
    while_loop = "while(false){\n    int i = 0;\n"
    if feature_types == "Most_Imp":
        #For most important features
        time_statement = "    Duration duration = Duration.between(startTime, endTime);\n    Datetime date=currentDate;\n"
    else:
        #For least important features
        time_statement = "    Maven maven = new Maven();\n    Random random = new Random();\n    Network network = new Network();\n    if (network.checkConnection()) {\n        maven.buildProject();\n    }\n"
    closing_brace = "}\n"
    code_to_add = while_loop + time_statement + closing_brace
    return code_to_add


def deadcode_perturbation_with_uc(feature_types="Most_Imp"):
    while_loop = "while(false){\n" 
    #uc_statement = "List<String> list = List<String>();\n list.add(""Class1"");\n list.add(""Class2"");\n Map<String, Integer> map = new HashMap<>();\n userAges.put(""Alice"", 30);\n JSON json = new JSON(""{city:""chicago"",name:""jon"");""\n assertEquals(""chicago"", jo.getString(""city""));\n Iterator<String> iterator = names.iterator();\n while(iterator.hasNext())\n{ String name = iterator.next();\n System.out.println(name);\n}\n"
    if feature_types == "Most_Imp":
        #For most important features
        uc_statement="    List<String> list = List<String>();\n    list.add(""Class1"");\n    list.add(""Class2"");\n    Map<String, Integer> dataMap = new HashMap<>();\n    dataMap.put(""age"", 22);\n    Assert.assertEquals(22, dataMap.get(""age""));\n    List<String> list = List<String>();\n    list.add(""Class1"");\n    list.add(""Class2"");\n    Iterator<String> iterator = names.iterator();\n    while(iterator.hasNext()) {\n        String name = iterator.next();\n        System.out.println(name);\n"
    else:
    #For least important features
        uc_statement = "    Multimap<String, String> context = new Multimap<>();\n    context.put(\"Scheme\", uri.getScheme());\n    Assertions.assertTrue(uri.getScheme().equals(\"abc\"), \"URI scheme is not 'abc'.\");\n"
    closing_brace = "    }\n}\n"
    code_to_add = while_loop + uc_statement + closing_brace
    return code_to_add


def deadcode_perturbation_with_od(feature_types="Most_Imp"):
    while_loop = "while(false){\n    int i = 0;\n"
    #od_statement = " dirString = names.lookup();\n PathData[] items = Path.getDirectoryPaths();\n  File file = new File(dirString, Path);\n assertEquals(file, ""Service.Store.Dir.fileName"");\n"

    if feature_types == "Most_Imp":
        #For most important features
        od_statement = "    ServiceContext context = new ServiceContext();\n    context.bind(""NameService"", ""NamingLookup"");\n    Path path = Mockito.mock(Path.class);\n    Mockito.when(path.toString()).thenReturn(""/mock/directory/test.txt"");\n    PathData[] pathDir = subtree.getDirectory();\n"
    else:
        #For least important features
        od_statement = "    Repository repo = new Repository();\n    repo.addCountryRecords(country, records);\n    System.out.println(\"Added \" + records + \" records for \" + country + \" to the repository.\");\n"
    closing_brace = "}\n"
    code_to_add = while_loop + od_statement + closing_brace
    return code_to_add

def deadcode_perturbation_with_nonflaky(feature_types="Most_Imp"):
    while_loop = "while(false){\n    int i = 0;\n"

    if feature_types == "Most_Imp":
        #For most important features
        nonflaky_statement = "    test = dirname.equals(\"Test\");\n    final String result = new Result();\n    Property put = result.put(\"Exception\");\n"
    else:
        #For least important features
        nonflaky_statement = "    specifiedGeo = ""USA"";\n    SearchLookupGeometry geo = Windows.MultipleLookupGeometry(params, view, compact);\n"
    closing_brace = "}\n"
    code_to_add = while_loop + nonflaky_statement + closing_brace
    return code_to_add

#def deadcode_perturbation_with_nonflaky():
#    while_loop = "while(false){\n int i = 0;\n"
#    nonflaky_statement = "";
#    \n"
#    closing_brace = "}\n"
#    code_to_add = while_loop + od_statement + closing_brace
#    return code_to_add
#********************************************************************
#********************************************************************

def generate_random_number(exclude_number):
    start = 0
    end = 5
    if exclude_number == 5:
       end = 3
    elif exclude_number == 0: 
        start = 1
    numbers = [num for num in range(start, end + 1) if num != exclude_number]
    perturb_cat = random.choice(numbers)
    #perturb_cat = 0
    #if exclude_number == 0: 
    #    return perturb_cat
    #else:
    #    perturb_cat = 0
    #    return perturb_cat

    return perturb_cat 
#from collections import Counter
#usage_counter = Counter()
#
#def generate_random_number(exclude_number, usage_counter, total_count):
#    # Define the possible numbers excluding the excluded number
#    numbers = [num for num in range(6) if num != exclude_number]
#    
#    # Ensure fair distribution by checking the usage counter
#    least_used_numbers = [num for num in numbers if usage_counter[num] < total_count // (len(numbers) + 1)]
#    
#    # If all remaining numbers are used equally, just pick from the available ones
#    if not least_used_numbers:
#        least_used_numbers = numbers
#
#    # Randomly select from the least used numbers
#    selected_number = random.choice(least_used_numbers)
#    
#    # Update the usage counter
#    usage_counter[selected_number] += 1
#
#    return selected_number

def count_perturb_type(type_of_perturbation, perturb_cat_counts, num_perturbation):
    print("\nCounts of each unique perturb_cat:")
    filename = "../results/"+num_perturbation+type_of_perturbation+'Perturbation_least_important_features.csv'
    # Open the CSV file for writing
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(["Original Label", "Perturbation Category", "Count"])
        for (original_label, perturb_cat), count in sorted(perturb_cat_counts.items()):
            #print(f"Original Label {original_label}, Perturbation Category {perturb_cat}: {count}")
            writer.writerow([original_label, perturb_cat, count])

def get_perturbed_data(test_code, code_to_add):
    last_closing_brace_idx = test_code.rfind('}')
    # Find the indentation level of the line where the closing brace is
    line_start_idx = test_code[:last_closing_brace_idx].rfind('\n') + 1
    #indentation = test_code[line_start_idx:last_closing_brace_idx].split('}')[0]
    # Add the same indentation to the code_to_add
    #indented_code_to_add = "\n" + indentation + code_to_add
    indentation = test_code[line_start_idx:last_closing_brace_idx].strip()
    # Add the same indentation to each line of the code_to_add
    indented_code_to_add = "\n".join([indentation + "    " + line for line in code_to_add.splitlines()])

    updated_code = test_code[:last_closing_brace_idx] + indented_code_to_add +"\n" +test_code[last_closing_brace_idx:]
    return updated_code

def printStatement_insertion(X_test, y_test, perturb_cats=None, num_perturbation="single", feature_types="Most_Imp"):
    updated_code = ""
    perturb_cat_counts = {}
    for idx, (test_code, label) in enumerate(zip(X_test, y_test)):
        #perturb_cat = generate_random_number(label) #the perturb category is other than this original label category

        if perturb_cats:
            perturb_cat = perturb_cats[idx] #To keep the same perturbation always; This is needed for combinations of multi-perturbations
        else:
            perturb_cat = generate_random_number(label) 

        key = (label, perturb_cat)
        if key in perturb_cat_counts:
            #print("key=", key)
            perturb_cat_counts[key] += 1
        else:
            perturb_cat_counts[key] = 1

        if perturb_cat == 0:
            code_to_add = printStatement_perturbation_with_async(feature_types)
            updated_code = get_perturbed_data(test_code, code_to_add)
        
        if perturb_cat == 1:
            code_to_add = printStatement_perturbation_with_conc(feature_types)
            #updated_code = test_code + "\n" + code_to_add
            #last_closing_brace_idx = test_code.rfind('}')
            updated_code = get_perturbed_data(test_code, code_to_add)
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add +"\n" +test_code[last_closing_brace_idx:]

        if perturb_cat == 2:
            code_to_add = printStatement_perturbation_with_time(feature_types)
            #updated_code = test_code + "\n" + code_to_add
            #last_closing_brace_idx = test_code.rfind('}')
            updated_code = get_perturbed_data(test_code, code_to_add)
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add +"\n" +test_code[last_closing_brace_idx:]

        if perturb_cat == 3:
            code_to_add = printStatement_perturbation_with_uc(feature_types)
            #updated_code = test_code + "\n" + code_to_add
            #last_closing_brace_idx = test_code.rfind('}')
            updated_code = get_perturbed_data(test_code, code_to_add)
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add +"\n" +test_code[last_closing_brace_idx:]

        if perturb_cat == 4:
            code_to_add = printStatement_perturbation_with_od(feature_types)
            #updated_code = test_code + "\n" + code_to_add
            #last_closing_brace_idx = test_code.rfind('}')
            updated_code = get_perturbed_data(test_code, code_to_add)
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add +"\n" +test_code[last_closing_brace_idx:]

        if perturb_cat == 5:
            code_to_add = printStatement_perturbation_with_nonflaky(feature_types)
            #updated_code = test_code + "\n" + code_to_add
            #last_closing_brace_idx = test_code.rfind('}')
            updated_code = get_perturbed_data(test_code, code_to_add)
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add +"\n" +test_code[last_closing_brace_idx:]

        X_test.iloc[idx] = updated_code 

    count_perturb_type("printStatement", perturb_cat_counts, num_perturbation)
    return X_test

def single_comment_insertion(X_test, y_test, perturb_cats=None, num_perturbation="single", feature_types="Most_Imp"):
    updated_code = ""
    perturb_cat_counts = {}
    total_count = len(X_test) * len(range(6))
    for idx, (test_code, label) in enumerate(zip(X_test, y_test)):
        #perturb_cat = generate_random_number(label, usage_counter, total_count) #the perturb category is other than this original label category
        perturb_cat = generate_random_number(label) #the perturb category is other than this original label category
        if perturb_cats:
            perturb_cat = perturb_cats[idx] #To keep the same perturbation always; This is needed for combinations of multi-perturbations
        else:
            perturb_cat = generate_random_number(label) 
         
        key = (label, perturb_cat)
        if key in perturb_cat_counts:
            perturb_cat_counts[key] += 1
        else:
            perturb_cat_counts[key] = 1
        #if perturb_cat in perturb_cat_counts:
        #    perturb_cat_counts[perturb_cat] += 1
        #else:
        #    perturb_cat_counts[perturb_cat] = 1

        #print('label=', str(label), 'pertub_cat=', str(perturb_cat)) 
        if perturb_cat == 0:
            code_to_add = singleLine_perturbation_with_async(feature_types)
            #last_closing_brace_idx = test_code.rfind('}')
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add +"\n" +test_code[last_closing_brace_idx:]
            updated_code = get_perturbed_data(test_code, code_to_add)
        
        if perturb_cat == 1:
            code_to_add = singleLine_perturbation_with_conc(feature_types)
            #updated_code = test_code + "\n" + code_to_add
            #last_closing_brace_idx = test_code.rfind('}')
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add +"\n" +test_code[last_closing_brace_idx:]
            updated_code = get_perturbed_data(test_code, code_to_add)

        if perturb_cat == 2:
            code_to_add = singleLine_perturbation_with_time(feature_types)
            #updated_code = test_code + "\n" + code_to_add
            #last_closing_brace_idx = test_code.rfind('}')
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add +"\n" +test_code[last_closing_brace_idx:]
            updated_code = get_perturbed_data(test_code, code_to_add)

        if perturb_cat == 3:
            code_to_add = singleLine_perturbation_with_uc(feature_types)
            #updated_code = test_code + "\n" + code_to_add
            #last_closing_brace_idx = test_code.rfind('}')
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add +"\n" +test_code[last_closing_brace_idx:]
            updated_code = get_perturbed_data(test_code, code_to_add)

        if perturb_cat == 4:
            code_to_add = singleLine_perturbation_with_od(feature_types)
            #updated_code = test_code + "\n" + code_to_add
            #last_closing_brace_idx = test_code.rfind('}')
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add +"\n" +test_code[last_closing_brace_idx:]
            updated_code = get_perturbed_data(test_code, code_to_add)

        if perturb_cat == 5:
            code_to_add = singleLine_perturbation_with_nonflaky(feature_types)
            #updated_code = test_code + "\n" + code_to_add
            #last_closing_brace_idx = test_code.rfind('}')
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add +"\n" +test_code[last_closing_brace_idx:]
            updated_code = get_perturbed_data(test_code, code_to_add)
        
        X_test.iloc[idx] = updated_code 

    count_perturb_type("single_comment", perturb_cat_counts, num_perturbation)
    #X_test_df['full_code'] = X_test
    return X_test


def multi_comment_insertion(X_test, y_test, perturb_cats=None, num_perturbation="single", feature_types="Most_Imp"):
    updated_code = ""
    perturb_cat_counts = {}
    total_count = len(X_test) * len(range(6))
    for idx, (test_code, label) in enumerate(zip(X_test, y_test)):
        if perturb_cats:
            perturb_cat = perturb_cats[idx] #To keep the same perturbation always; This is needed for combinations of multi-perturbations
        else:
            perturb_cat = generate_random_number(label) 
        #perturb_cat = generate_random_number(label) #the perturb category is other than this original label category
        
        key = (label, perturb_cat)
        if key in perturb_cat_counts:
            perturb_cat_counts[key] += 1
        else:
            perturb_cat_counts[key] = 1
        #if perturb_cat in perturb_cat_counts:
        #    perturb_cat_counts[perturb_cat] += 1
        #else:
        #    perturb_cat_counts[perturb_cat] = 1
        
        #print('label=', str(label), 'pertub_cat=', str(perturb_cat) ) 
        if perturb_cat == 0:
            code_to_add = multiLine_perturbation_with_async(feature_types)
            #last_closing_brace_idx = test_code.rfind('}')
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add +"\n" +test_code[last_closing_brace_idx:]
            updated_code = get_perturbed_data(test_code, code_to_add)
        
        if perturb_cat == 1:
            code_to_add = multiLine_perturbation_with_conc(feature_types)
            #updated_code = test_code + "\n" + code_to_add
            #last_closing_brace_idx = test_code.rfind('}')
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add +"\n" +test_code[last_closing_brace_idx:]
            updated_code = get_perturbed_data(test_code, code_to_add)

        if perturb_cat == 2:
            code_to_add = multiLine_perturbation_with_time(feature_types)
            #updated_code = test_code + "\n" + code_to_add
            #last_closing_brace_idx = test_code.rfind('}')
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add +"\n" +test_code[last_closing_brace_idx:]
            updated_code = get_perturbed_data(test_code, code_to_add)

        if perturb_cat == 3:
            code_to_add = multiLine_perturbation_with_uc(feature_types)
            #updated_code = test_code + "\n" + code_to_add
            #last_closing_brace_idx = test_code.rfind('}')
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add +"\n" +test_code[last_closing_brace_idx:]
            updated_code = get_perturbed_data(test_code, code_to_add)

        if perturb_cat == 4:
            code_to_add = multiLine_perturbation_with_od(feature_types)
            #updated_code = test_code + "\n" + code_to_add
            #last_closing_brace_idx = test_code.rfind('}')
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add +"\n" +test_code[last_closing_brace_idx:]
            updated_code = get_perturbed_data(test_code, code_to_add)

        if perturb_cat == 5:
            code_to_add = multiLine_perturbation_with_nonflaky(feature_types)
            #updated_code = test_code + "\n" + code_to_add
            #last_closing_brace_idx = test_code.rfind('}')
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add +"\n" +test_code[last_closing_brace_idx:]
            updated_code = get_perturbed_data(test_code, code_to_add)
        
        X_test.iloc[idx] = updated_code 

    count_perturb_type("multi_comment", perturb_cat_counts, num_perturbation)
    return X_test



def deadcode_insertion(X_test, y_test, perturb_cats=None, num_perturbation="single", feature_types="Most_Imp"):
    updated_code = ""
    perturb_cat_counts = {}
    for idx, (test_code, label) in enumerate(zip(X_test, y_test)):
        if perturb_cats:
            print('perturb_cats=',perturb_cats)
            perturb_cat = perturb_cats[idx] #To keep the same perturbation always; This is needed for combinations of multi-perturbations
        else:
            perturb_cat = generate_random_number(label) 

        perturb_cat = generate_random_number(label) #the perturb category is other than this original label category
        
        key = (label, perturb_cat)
        if key in perturb_cat_counts:
            perturb_cat_counts[key] += 1
        else:
            perturb_cat_counts[key] = 1
        
        if perturb_cat == 0: #Async
            code_to_add = deadcode_perturbation_with_async(feature_types)
            print('*** From perturbation ******; feature_types=', feature_types)
            #print(code_to_add)
            #print('test_code=')
            #print(test_code)
            #updated_code = test_code + "\n" + code_to_add
            #last_closing_brace_idx = test_code.rfind('}')
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add + test_code[last_closing_brace_idx:]
            updated_code = get_perturbed_data(test_code, code_to_add)
            #print(type(updated_code))
            #print(updated_code)
            X_test.iloc[idx] = updated_code
        elif perturb_cat == 1: #Concurency 
            code_to_add = deadcode_perturbation_with_conc(feature_types)
            #updated_code = test_code + "\n" + code_to_add
            #last_closing_brace_idx = test_code.rfind('}')
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add + test_code[last_closing_brace_idx:]
            updated_code = get_perturbed_data(test_code, code_to_add)
            X_test.iloc[idx] = updated_code
        elif perturb_cat == 2: # Time
            code_to_add = deadcode_perturbation_with_time(feature_types)
            #updated_code = test_code + "\n" + code_to_add
            #last_closing_brace_idx = test_code.rfind('}')
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add + test_code[last_closing_brace_idx:]
            updated_code = get_perturbed_data(test_code, code_to_add)
            X_test.iloc[idx] = updated_code 

        if perturb_cat == 3:
            code_to_add = deadcode_perturbation_with_uc(feature_types)
            #updated_code = test_code + "\n" + code_to_add
            #last_closing_brace_idx = test_code.rfind('}')
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add + test_code[last_closing_brace_idx:]
            updated_code = get_perturbed_data(test_code, code_to_add)
            X_test.iloc[idx] = updated_code

        if perturb_cat == 4:
            code_to_add = deadcode_perturbation_with_od(feature_types)
            #updated_code = code + "\n" + code_to_add
            #last_closing_brace_idx = test_code.rfind('}')
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add + test_code[last_closing_brace_idx:]
            updated_code = get_perturbed_data(test_code, code_to_add)
            X_test.iloc[idx] = updated_code 
       
        elif perturb_cat == 5:
            code_to_add = deadcode_perturbation_with_nonflaky(feature_types)
            #last_closing_brace_idx = test_code.rfind('}')
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add + test_code[last_closing_brace_idx:]
            updated_code = get_perturbed_data(test_code, code_to_add)
            X_test.iloc[idx] = updated_code

    count_perturb_type("deadcode", perturb_cat_counts, num_perturbation)
    return X_test


def variableRenaming_insertion(X_test, y_test, perturb_cats=None, num_perturbation="single", feature_types="Most_Imp"):
    updated_code = ""
    perturb_cat_counts = {}
    for idx, (test_code, label) in enumerate(zip(X_test, y_test)):
        #perturb_cat = generate_random_number(label) #the perturb category is other than this original label category
        if perturb_cats:
            perturb_cat = perturb_cats[idx] #To keep the same perturbation always; This is needed for combinations of multi-perturbations
        else:
            perturb_cat = generate_random_number(label) 

        key = (label, perturb_cat)
        if key in perturb_cat_counts:
            perturb_cat_counts[key] += 1
        else:
            perturb_cat_counts[key] = 1

        '''if perturb_cat in perturb_cat_counts:
            perturb_cat_counts[perturb_cat] += 1
        else:
            perturb_cat_counts[perturb_cat] = 1'''
        #print('*********perturb_cat=',perturb_cat, ", feature_types=", feature_types, ",label=", label)
        if perturb_cat == 0:
            if feature_types == "Most_Imp":
                variable_name_list = ["sleep", "await", "future", "unit", "poll", "connection", "wait",  "is", "service", "thread", "future", "prepare", "run", "events", "test", "uri", "servlet", "workflow", "duration", "point", "now", "point", "mapper", "worker", "block", "ms", "yield", "iio", "abc", "ppo", "xyz", "ylk", "oow", "lksm", "wer", "qwe", "rek", "llinePath", "wqr", "mlk", "sst", "qwa", "dfg", "llq", "aas", "zza", "bba", "dda", "ssa", "wwp", "ggf", "lt", "le", "lw", "lx", "lc", "sleep1", "wait1", "thread1", "thread2", "run3"] 
            else:
                #Least Impact tokens
                #variable_name_list = ["snapshot", "query", "ttl","jar", "me", "control", "greeter", "validator", "read", "run1", "capture", "dispatch", "acked", "sex", "ums", "ums", "cursor", "message", "ledger", "ops", "operator", "spy", "report", "crud", "health", "task", "ms", "that", "t", "region", "ppo", "tomcat", "ylk", "oow", "lksm", "wer", "qwe", "rek", "llinePath", "wqr", "mlk", "sst", "qa", "qw", "qr", "qt", "qo", "qk", "qo", "qp", "qm" , "lt", "le", "lw", "lx", "lc"]
                variable_name_list = ["upgrade", "meta", "crash"]
            updated_code = variableRenaming_perturbation(test_code, idx, variable_name_list, perturb_cat)
            #last_closing_brace_idx = test_code.rfind('}')
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add +"\n" +test_code[last_closing_brace_idx:]
        
        if perturb_cat == 1:
            if feature_types == "Most_Imp":
                variable_name_list = ["Duration", "concurrenct", "wait", "automic", "latch", "interrupted", "schedule", "sleep", "duration", "bolt", "manager", "atomic", "distributed", "write", "robust", "matcher", "choke", "stub", "hello","semaphore", "iio", "abc", "ppo", "xyz", "ylk", "oow", "lksm", "wer", "qwe", "rek", "llinePath", "wqr", "mlk", "sst", "qa", "qw", "qr", "qt", "qo", "qk", "qo", "qp", "qm" , "lt", "le", "lw", "lx", "lc"]
            else:
                #Least Impact tokens
                #variable_name_list = ["runtime", "bulk", "down", "last", "consumer", "operation", "subject", "verify", "worker", "batch", "failed", "stlset", "zkw", "adder", "latch", "corrupt", "raided", "hello","executed", "pool", "grace", "kafka", "xyz", "ylk", "oow", "lksm", "wer", "qwe", "rek", "llinePath", "wqr", "mlk", "sst", "qa", "qw", "qr", "qt", "qo", "qk", "qo", "qp", "qm" , "lt", "le", "lw", "lx", "lc"]
                variable_name_list = ["mode", "cluster", "tokens"]
            updated_code = variableRenaming_perturbation(test_code, idx, variable_name_list, perturb_cat)
            #code_to_add = printStatement_perturbation_with_conc()
            ##updated_code = test_code + "\n" + code_to_add
            #last_closing_brace_idx = test_code.rfind('}')
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add +"\n" +test_code[last_closing_brace_idx:]

        if perturb_cat == 2:
            if feature_types == "Most_Imp":
                variable_name_list = ["time", "timestamp", "millis", "now", "duration", "clock", "seconds", "builder", "min", "month", "snapshot", "service", "offset", "response", "satisfy", "update", "live", "year", "current", "nano", "days", "hours", "yesterday", "absent",  "minutes", "duration2", "timer2", "isInstant", "timestamp", "optional", "handle", "equ", "fn", "iio", "abc", "ppo", "xyz", "ylk", "oow", "lksm", "wer", "qwe", "rek", "llinePath", "wqr", "mlk", "sst",  "date1",  "date2", "date3", "date5", "day2", "day3", "day10", "time", "date1", "date2", "date3", "date4", "date5", "date6", "ppl", "lkw", "qwa", "dfg", "lt", "le", "lw", "lx", "lc"]
            else:
                #Least Impact tokens
                #variable_name_list = ["checkpoint", "vo", "greater", "etlbatch", "yield", "through", "metadata", "yesterday", "new", "set", "updated", "mtime1", "suffix", "exporter", "start", "modify", "response", "execution", "snapshot", "expected", "pm", "parser", "instant", "delete",  "distributer", "instant2", "instant2",  "optional", "handle", "equ", "fn", "iio", "abc", "ppo", "xyz", "ylk", "oow", "lksm", "wer", "qwe", "rek", "llinePath", "wqr", "mlk", "sst",  "date1",  "date2", "date3", "date5", "day2", "day3", "day10", "time", "date1", "date2", "date3", "date4", "date5", "date6", "ppl", "lkw", "qwa", "dfg", "lt", "le", "lw", "lx", "lc"]
                variable_name_list = ["left", "compare", "max"]
            updated_code = variableRenaming_perturbation(test_code, idx, variable_name_list, perturb_cat)
            #code_to_add = printStatement_perturbation_with_time()
            ##updated_code = test_code + "\n" + code_to_add
            #last_closing_brace_idx = test_code.rfind('}')
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add +"\n" +test_code[last_closing_brace_idx:]

        if perturb_cat == 3:
            if feature_types == "Most_Imp":
                variable_name_list = ["set", "sample", "hash", "list", "iterator","map", "dictionary", "hashmap", "hash", "linkedlist", "stack", "array", "iterator", "class", "hash", "parse", "str", "fail", "system", "metadata", "pairs", "event", "iio", "abc", "ppo", "xyz", "ylk", "oow", "lksm", "wer", "qwe", "rek", "llinePath", "wqr", "mlk", "sst", "qwa", "dfg", "llq", "aas", "zza", "bba", "dda", "ssa", "wwp", "ggf" , "qa", "qw", "qr", "qt", "qo", "qk", "qo", "qp", "qm", "lt", "le", "lw", "lx", "lc", "stack", "vector", "enumMap", "hashTable", "treeSet", "set1", "map1", "collection1", "list2", "json2", "dict2"]
            else:
                #Least Impact tokens
                #variable_name_list = ["utils", "reference", "devs", "add", "c2", "order", "one", "case", "as", "first", "metrics", "this", "hash", "object", "twin", "unirest", "full", "dubbo", "foo", "cc", "iio", "abc", "ppo", "xyz", "ylk", "oow", "lksm", "wer", "qwe", "rek", "llinePath", "wqr", "mlk", "sst", "qwa", "dfg", "llq", "aas", "zza", "bba", "dda", "ssa", "wwp", "ggf" , "qa", "qw", "qr", "qt", "qo", "qk", "qo", "qp", "qm", "lt", "le", "lw", "lx", "lc"]
                variable_name_list = ["value", "parse", "partition"]
            updated_code = variableRenaming_perturbation(test_code, idx, variable_name_list, perturb_cat)
            #code_to_add = printStatement_perturbation_with_uc()
            ##updated_code = test_code + "\n" + code_to_add
            #last_closing_brace_idx = test_code.rfind('}')
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add +"\n" +test_code[last_closing_brace_idx:]

        if perturb_cat == 4:
            if feature_types == "Most_Imp":
                variable_name_list = ["name", "file", "path", "create", "lookup", "uri", "store", "context", "item", "system", "object", "dir", "service", "directory", "action", "loopup", "store", "latch", "directory", "filePath", "dirPath", "clusterPath", "staticObj", "ip", "sharedKey", "model", "iio", "abc", "ppo", "xyz", "ylk", "oow", "lksm", "wer", "qwe", "rek", "llinePath", "wqr", "mlk", "sst", "lt", "le", "lw", "lx", "lc"]
            else:
                #Least Impact tokens   
                #variable_name_list = ["protocol", "signature", "version", "that", "is", "d1", "wildfly", "rejections", "remain", "mode", "mojo", "analyze", "t", "referenceable", "permissions", "exists", "sleep", "empty", "based", "skip", "found", "sentence", "iio", "abc", "ppo", "xyz", "ylk", "oow", "lksm", "wer", "qwe", "rek", "llinePath", "wqr", "mlk", "sst", "lt", "le", "lw", "lx", "lc"]
                variable_name_list = ["all", "ubbo", "remote"]
            updated_code = variableRenaming_perturbation(test_code, idx, variable_name_list, perturb_cat)
            #code_to_add = printStatement_perturbation_with_od()
            ##updated_code = test_code + "\n" + code_to_add
            #last_closing_brace_idx = test_code.rfind('}')
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add +"\n" +test_code[last_closing_brace_idx:]

        elif perturb_cat == 5:
            if feature_types == "Most_Imp":
                variable_name_list = ["equals", "successful", "result", "foo", "trial" , "object", "count", "booleans", "default", "my", "wrapper", "clear", "field", "tor", "serialize", "test", "abc", "xyz", "ppo", "llp", "mkn", "hgh", "oow", "iio", "abc", "ppo", "xyz", "ylk", "oow", "lksm", "wer", "qwe", "rek", "llinePath", "wqr", "mlk", "sst", "trial2", "trail3", "trial4", "trial5", "iiw", "ppl", "lkw", "qwa", "dfg", "llq", "aas", "zza", "bba", "dda", "ssa", "wwp", "ggf", "qa", "qw", "qr", "qt", "qo", "qk", "qo", "qp", "qm", "lk", "lp", "lm", "lh", "lt", "le", "lw", "lx", "lc", "zx", "zc", "zv", "zb", "zn", "zm"]
            else:
                #Least Impact tokens
                #variable_name_list = ["mode", "unit", "only", "constant", "i" , "res", "set", "from", "generator", "return", "sharing", "remove", "disable", "res", "remove", "until", "abc", "xyz", "ppo", "llp", "mkn", "hgh", "oow", "iio", "abc", "ppo", "xyz", "ylk", "oow", "lksm", "wer", "qwe", "rek", "llinePath", "wqr", "mlk", "sst", "trial2", "trail3", "trial4", "trial5", "iiw", "ppl", "lkw", "qwa", "dfg", "llq", "aas", "zza", "bba", "dda", "ssa", "wwp", "ggf", "qa", "qw", "qr", "qt", "qo", "qk", "qo", "qp", "qm", "lk", "lp", "lm", "lh", "lt", "le", "lw", "lx", "lc", "zx", "zc", "zv", "zb", "zn", "zm"]
                variable_name_list = ["izard", "course", "violation"]
                #variable_name_list = []
#{"sleep", ""}
            updated_code = variableRenaming_perturbation(test_code, idx, variable_name_list, perturb_cat)
            #code_to_add = printStatement_perturbation_with_nonflaky()
            ##updated_code = test_code + "\n" + code_to_add
            #last_closing_brace_idx = test_code.rfind('}')
            #updated_code = test_code[:last_closing_brace_idx] + code_to_add +"\n" +test_code[last_closing_brace_idx:]

        #if category != 0 or category !=1: #Ignoring async category to add thread.sleep
        #    code_to_add = printStatement_other_than_asyncAnd_concurrency_cat()
        #    updated_code = code + "\n" + code_to_add
        #elif category != 2:
        #    code_to_add = printStatement_other_than_time_cat()
        #    updated_code = code + "\n" + code_to_add
        X_test.iloc[idx] = updated_code 

    #print("\nCounts of each unique perturb_cat:")
    #for cat in sorted(perturb_cat_counts):
    #    print(f"perturb_cat {cat}: {perturb_cat_counts[cat]}")
    #X_test_df['full_code'] = X_test
    count_perturb_type("variableRenaming", perturb_cat_counts, num_perturbation)
    print("X_test after variableRenaming_insertion:", X_test.head())
    return X_test


