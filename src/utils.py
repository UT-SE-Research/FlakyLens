import os
import logging
import numpy as np
import torch
import random
from transformers import AdamW, AutoTokenizer, AutoModel, AutoConfig, T5Tokenizer, T5ForConditionalGeneration, T5EncoderModel, RobertaTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import json
import pandas as pd
from captum.attr import LayerIntegratedGradients, LLMGradientAttribution
import textwrap

#from huggingface_hub import login

def make_few_show_example(fold_number):
    #print("fold_number=", fold_number)
    if fold_number <= 3:
        examples=textwrap.dedent("""\
    Example 1: 
    Test:
    @Test
    public void testMenuInvalidationAfterDestroy() throws Throwable {
        final A activity = getActivity();
        getInstrumentation().runOnMainSync(new Runnable() {
            @Override
            public void run() {
                activity.reset();
                assertNull(activity.getMenu());
                activity.supportInvalidateOptionsMenu();
                getInstrumentation().callActivityOnDestroy(activity);
            }
        });
        Thread.sleep(100);
        assertNull(activity.getMenu());
    }
    ```
    **Category**: async wait
    ```

    Example 2: 
    Test:
    @Test
    public void serverStatusUnthrottledIfJournalUtilizationIsLowerThanThreshold() throws Exception {
        serverStatus.throttle();
        final Size segmentSize = Size.kilobytes(1L);
        final KafkaJournal journal = new KafkaJournal(journalDirectory, scheduler, segmentSize, Duration.standardSeconds(1L), Size.kilobytes(4L), Duration.standardSeconds(1L), 1000000, Duration.standardSeconds(1L), 90, new MetricRegistry(), serverStatus);
        journal.flushDirtyLogs();
        journal.cleanupLogs();
        assertThat(serverStatus.getLifecycle()).isEqualTo(RUNNING);
    }
    ```
    **Category**: concurrency
    ```

    Example 3: 
    Test:
    @Test
    public void getNanoTime() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            long time = mRecyclerView.getNanoTime();
            assertNotEquals(0, time);
            assertNotEquals(time, mRecyclerView.getNanoTime());
        } else {
            assertEquals(0, mRecyclerView.getNanoTime());
        }
    }
    ```
    **Category**: time  
    ```
     
    Example 4: 
    Test:
    @Test
    public void findingAnnotationsRecursively() {
        Method method = findMethod(this.getClass(), ""withMarker"", String.class);
        List<Annotation> annotations = allAnnotations(method.getParameters()[0]);
        assertEquals(4, annotations.size());
        assertEquals(X.class, annotations.get(0).annotationType());
        assertEquals(Y.class, annotations.get(1).annotationType());
        assertEquals(Z.class, annotations.get(2).annotationType());
        assertEquals(W.class, annotations.get(3).annotationType());
    }
    ```
    **Category**: unordered collections
    ```
    
    Example 5: 
    Test:
    @Test
    public void testCompositeBindingOps() throws Exception {
        final KernelServices services = createKernelServicesBuilder(createAdditionalInitialization()).setSubsystemXml(getSubsystemXml()).build();
        final ModelNode addr = Operations.createAddress(ModelDescriptionConstants.SUBSYSTEM, NamingExtension.SUBSYSTEM_NAME, NamingSubsystemModel.BINDING, ""java:global/alookup"");
        final ModelNode addOp = Operations.createAddOperation(addr);
        addOp.get(NamingSubsystemModel.BINDING_TYPE).set(NamingSubsystemModel.LOOKUP);
        final ModelNode compositeOp = Operations.CompositeOperationBuilder.create().addStep(addOp).addStep(Operations.createWriteAttributeOperation(addr, NamingSubsystemModel.LOOKUP, ""java:global/a"")).build().getOperation();
        ModelTestUtils.checkOutcome(services.executeOperation(compositeOp));
    }
    ```
    **Category**: order dependent test
    ```

    Example 6: 
    Test:
    @Test 
    public void testInstanceBinding() throws Exception {
        			public Date bindDate() {
        				return date;
        			}
    ```
    **Category**: not flaky
    ```
    """)
    elif fold_number == 4:
        examples=textwrap.dedent("""\
        Example 1: 
        Test:
        @Test
        public void shouldOnlyHandleRequestsOfSubscribedTypes() {
            serverTransport.subscribe(0, COMMAND, new DirectlyResponder());
            serverTransport.subscribe(0, UNKNOWN, new FailingResponder());
            final var requestFuture = clientTransport.sendRequest(() -> AtomixTransportTest.serverAddress, new Request(""messageABC""), REQUEST_TIMEOUT);
            final var response = requestFuture.join();
            assertThat(response.byteArray()).isEqualTo(""messageABC"".getBytes());
        }
        **Category**: async wait

        Example 2: 
        Test:
        @Test
        public void shouldBeAbleToCancelJob() throws Exception {
            RobustJobSchedulerWrapper robustWrapper = new RobustJobSchedulerWrapper(actualScheduler, log);
            AtomicInteger count = new AtomicInteger();
            JobHandle jobHandle = robustWrapper.scheduleRecurring(""JobName"", 1, count::incrementAndGet);
            assertEventually(""run count"", count::get, Matchers.greaterThanOrEqualTo(100), DEFAULT_TIMEOUT_MS, MILLISECONDS);
            robustWrapper.cancelAndWaitTermination(jobHandle);
            int finalCount = count.get();
            Thread.sleep(50);
            assertEquals(finalCount, count.get());
        }
        **Category**: concurrency
        
        Example 3: 
        Test:
        @Test
        public void testBacklogLimiter() {
            long duration = runWithRate(2 * RateLimiting.DEFAULT_MAX_PARALLELISM,-1.0 , new DelayFn<Integer>());
            Assert.assertThat(duration,greaterThan(2 * DelayFn.DELAY_MS));
         }
         **Category**: time  
         
        Example 4: 
        Test:
        @Test
        public void getFieldNamesTest() {
            List<String> names = EnumUtil.getFieldNames(TestEnum.class);
            Assert.assertEquals(CollUtil.newArrayList(""type"", ""name""), names);
        }
        **Category**: unordered collections
    
        Example 5: 
        Test:
        @Test
        public void canHandleUnfragmentedMessageReceived() {
            NMEAMessage unfragmentedNMEAMessage = NMEAMessage.fromString(""!AIVDM,1,1,,B,15MqdBP000G@qoLEi69PVGaN0D0=,0*3A"");
            final ArgumentCaptor<AISMessage> aisMessage = new ArgumentCaptor<>();
            context.checking(new Expectations() {{
                oneOf(aisMessageHandler).accept(with(aisMessage.getMatcher()));
            }});
            aisMessageReceiver.accept(unfragmentedNMEAMessage);
            assertEquals(AISMessageType.PositionReportClassAScheduled, aisMessage.getCapturedObject().getMessageType());
        }
        **Category**: order dependent test
        
        Example 6: 
        Test:
        @Test 
        public void testUpdatedCurrentPeakSizeLogsOnlyOnChange() throws Exception {
            StringLogger logger = mock( StringLogger.class );
            LoggingResourcePoolMonitor monitor = new LoggingResourcePoolMonitor( logger );
    
            monitor.updatedCurrentPeakSize( 10 );
            verify( logger, times( 1 ) ).debug( anyString() );
    
            monitor.updatedCurrentPeakSize( 10 );
            verify( logger, times( 1 ) ).debug( anyString() );
    
            monitor.updatedCurrentPeakSize( 11 );
            verify( logger, times( 2 ) ).debug( anyString() );
        }
        **Category**: not flaky""")
       
    return examples

# Function to check if a token contains at least one English letter
def contains_english_letter(token):
    return bool(re.search(r'[a-zA-Z]', token))

def filter_tokens(sorted_token_attributions):
    # Remove 'Ġ' prefix and filter out non-English tokens
    filtered_token_attributions = []
    unique_tokens = set()  

    for token, score in sorted_token_attributions:
        cleaned_token = token.lstrip("Ġ").lstrip(".").lstrip("_") 
        
        # Ensure the token contains at least one English letter AND is longer than 1 character
        if contains_english_letter(cleaned_token) and len(cleaned_token) > 1:
            if cleaned_token not in unique_tokens:  #Ensure uniqueness
                unique_tokens.add(cleaned_token)
                filtered_token_attributions.append((cleaned_token, score))
 
            if len(unique_tokens) == 20:
                break
    # Print filtered top-20 tokens
    print("\n Filtered Top-20 Tokens (Only English Letters, No 'Ġ' Prefix):")
    for token, score in filtered_token_attributions:  # Get only top 20 tokens
        print(f"Token: {token}, Attribution Score: {score:.4f}")
    
    top_20_tokens = [token for token, _ in filtered_token_attributions[:20]]

    # If you need to return the cleaned list:
    return top_20_tokens

def forward_func(inputs, attention_mask, model):
    """Forward function to return logits for attribution."""
    outputs = model(inputs, attention_mask=attention_mask, return_dict=True)
    return outputs.logits[:, -1, :]  # Get logits for the last token


def predict(inputs, attention_mask, model, target_token):
    """Forward pass for IG. Returns logits for classification."""
    outputs = model(inputs.long(), attention_mask=attention_mask, return_dict=True)
    return outputs.logits[:, -1, :] #outputs.logits[:, -1, :].max(dim=-1).values  # Return logits for last token

stopwords = {"<|im_start|>", "<|im_end|>", ";", "()", "=", "(", ")", ".", ",", "Ċ", "Ġ", "Ġ<", "Ġ>" , "void", "_" }

def clean_token(token):
    return re.sub(r"^[^a-zA-Z]+", "", token)  # Remove leading non-letters


def interpret_with_ig_codellama(prompt, tokenizer, model, model_inputs, predicted_tokens, test_data):
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]  #Ensure this is defined

    lig = LayerIntegratedGradients(forward_func, model.model.embed_tokens)
    
    # Convert Input IDs to Tokenized Text
    tokenized_prompt = tokenizer.convert_ids_to_tokens(input_ids[0])
    print('tokenized_prompt=', tokenized_prompt)

    # Locate the "Test:" marker in tokenized input
    test_start_index = -1
    for i in range(len(tokenized_prompt) - 1):  # No need for -2
        if tokenized_prompt[i] == '▁Test' and tokenized_prompt[i + 1] == ':':
            test_start_index = i + 2  # Move past "Test:"
            break

    if test_start_index == -1: 
        print("Error: 'Test:' not found in tokenized input! Computing for all tokens.")
        test_start_index = 0  # Default to full prompt
        test_end_index = len(tokenized_prompt)
    else:
        # Locate "@Test"
        for i in range(test_start_index, len(tokenized_prompt) - 1): 
            if tokenized_prompt[i] == '▁@' and tokenized_prompt[i + 1] == 'Test':
                test_start_index = i  # Adjust start index
                break

        # Find the end of the test method (closing '}')
        test_end_index = -1
        for i in range(test_start_index, len(tokenized_prompt)):
            if tokenized_prompt[i] == '▁}':  # Match tokenized version
                test_end_index = i + 1  # Include '}'
                break

        if test_end_index == -1: 
            test_end_index = len(tokenized_prompt)  # If no '}', take full remaining tokens

    print(f"Test starts at token index: {test_start_index}, ends at: {test_end_index}")
    MAX_TEST_LENGTH = 1024  # Truncate large test functions
    if test_end_index - test_start_index < MAX_TEST_LENGTH:
        if len(predicted_tokens.shape) > 1:
            target_token = predicted_tokens[0, -1].item()  # Last token
        else:
            target_token = predicted_tokens.item()  # Scalar

        attributions = lig.attribute(input_ids, target=target_token,     additional_forward_args=(attention_mask, model), n_steps=1)
            
        # Normalize attributions for better visualization
        attributions = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
        attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min() + 1e-6)
        # Get the tokens and attributions only within the test_code range
        filtered_tokens = tokenized_prompt[test_start_index:test_end_index]
        filtered_tokens = [clean_token(token) for token in filtered_tokens]  # Clean tokens 
        filtered_attributions = attributions[test_start_index:test_end_index]

        filtered_tokens_attributions = [(token, attribution) for token, attribution in zip(filtered_tokens, filtered_attributions) if token not in stopwords]

        # Sort tokens based on attributions
        #sorted_tokens = sorted(zip(filtered_tokens_attributions, attributions), key=lambda x: x[1], reverse=True)
        sorted_tokens = sorted(filtered_tokens_attributions, key=lambda x: x[1], reverse=True)
        # Free up memory
        del attributions
        torch.cuda.empty_cache()
    else:
        sorted_tokens = []
    return sorted_tokens 


def interpret_with_ig_gemma7b(prompt, tokenizer, model, model_inputs, predicted_tokens, test_data):
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]  # Ensure this is defined

    lig = LayerIntegratedGradients(forward_func, model.model.embed_tokens)
    # Identify Token Indices for test_data (Test Code)
    tokenized_prompt = tokenizer.convert_ids_to_tokens(input_ids[0])
    tokenized_test_code = tokenizer.tokenize(test_data)  # Tokenize test_code separately

    # Convert Input IDs to Tokenized Text
    tokenized_prompt = tokenizer.convert_ids_to_tokens(input_ids[0])
    print('tokenized_prompt=', tokenized_prompt)
    #exit()
    # Locate the "Test:" marker in tokenized input
    test_start_index = -1
    for i in range(len(tokenized_prompt) - 2): 
        if tokenized_prompt[i] == 'Test' and (tokenized_prompt[i + 1] == ':'):
            test_start_index = i + 2  # Move past "Test:"
            break

    if test_start_index == -1: 
        print(" Error: 'Test:' not found in tokenized input! Computing for all tokens.")
        test_start_index = 0  # Default to full prompt
        test_end_index = len(tokenized_prompt)
    else:
        # Locate "@Test"
        for i in range(test_start_index, len(tokenized_prompt) - 1): 
            if tokenized_prompt[i] == '@' and tokenized_prompt[i + 1] == 'Test':
                test_start_index = i  # Adjust start index
                break

        # Find the end of the test method (closing '}')
        test_end_index = -1
        for i in range(test_start_index, len(tokenized_prompt)):
            if tokenized_prompt[i] == '}':
                test_end_index = i + 1  # Include '}'
                break

        if test_end_index == -1: 
            test_end_index = len(tokenized_prompt)  # If no '}', take full remaining tokens

    filtered_tokens_attributions = []
    MAX_TEST_LENGTH = 1024  # Truncate large test functions
    print('test_end_index - test_start_index=',test_end_index - test_start_index)
    #exit()
    if test_end_index - test_start_index <= MAX_TEST_LENGTH:
        attributions = lig.attribute(input_ids, target=predicted_tokens[0, 0].item(),     additional_forward_args=(attention_mask, model), n_steps=1)
            
        attributions = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
        attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min() + 1e-6)
        #  Get the tokens and attributions only within the test_code range
        filtered_tokens = tokenized_prompt[test_start_index:test_end_index]
        filtered_tokens = [clean_token(token) for token in filtered_tokens]  # Clean tokens 
        filtered_attributions = attributions[test_start_index:test_end_index]

        filtered_tokens_attributions = [(token, attribution) for token, attribution in zip(filtered_tokens, filtered_attributions) if token not in stopwords]

        sorted_tokens = sorted(filtered_tokens_attributions, key=lambda x: x[1], reverse=True)
        del attributions
        torch.cuda.empty_cache()
    else:
        sorted_tokens = []
    return sorted_tokens 


import gc
def interpret_with_ig_qwen(prompt, tokenizer, model, model_inputs, predicted_tokens, test_data, ml_technique):

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]  # Ensure this is defined
    if input_ids.shape[0] > 1:
        print('multiple examples are runnning.....')
    if ml_technique == "deep_seek_coder" or ml_technique == "llama3_8b" or ml_technique == "qwen":
        MAX_TOKENS = 512  # Reduce to 512 tokens
        input_ids = input_ids[:, :MAX_TOKENS]
        attention_mask = attention_mask[:, :MAX_TOKENS]
 
    lig = LayerIntegratedGradients(forward_func, model.model.embed_tokens)
    # Identify Token Indices for test_data (Test Code)
    tokenized_prompt = tokenizer.convert_ids_to_tokens(input_ids[0])
    tokenized_test_code = tokenizer.tokenize(test_data)  # Tokenize test_code separately

    # Convert Input IDs to Tokenized Text
    tokenized_prompt = tokenizer.convert_ids_to_tokens(input_ids[0])
    print('tokenized_prompt=', tokenized_prompt)
    #exit()
    # Locate the "Test:" marker in tokenized input
    test_start_index = -1
    for i in range(len(tokenized_prompt) - 2): 
        if tokenized_prompt[i] == 'ĠTest' and (tokenized_prompt[i + 1] == ':' or tokenized_prompt[i + 1] == ':Ċ'):
            test_start_index = i + 2  # Move past "Test:"
            break

    if test_start_index == -1: 
        print(" Error: 'Test:' not found in tokenized input! Computing for all tokens.")
        test_start_index = 0  # Default to full prompt
        test_end_index = len(tokenized_prompt)
    else:
        # Locate "@Test"
        for i in range(test_start_index, len(tokenized_prompt) - 1): 
            if tokenized_prompt[i] == 'Ġ@' and tokenized_prompt[i + 1] == 'Test':
                test_start_index = i  # Adjust start index
                break

        # Find the end of the test method (closing '}')
        test_end_index = -1
        for i in range(test_start_index, len(tokenized_prompt)):
            if tokenized_prompt[i] == '}':
                test_end_index = i + 1  # Include '}'
                break

        if test_end_index == -1: 
            test_end_index = len(tokenized_prompt)  # If no '}', take full remaining tokens

    print(f" Test starts at token index: {test_start_index}, ends at: {test_end_index}")
    filtered_tokens_attributions = []
    MAX_TEST_LENGTH = 1200  # Truncate large test functions
    print('test_end_index - test_start_index=',test_end_index - test_start_index)
    #exit()
    if test_end_index - test_start_index <= MAX_TEST_LENGTH:
        torch.cuda.empty_cache()  # Free unused GPU memory
        gc.collect()
        with torch.cuda.amp.autocast():
            attributions = lig.attribute(input_ids, target=predicted_tokens[0, 0].item(),     additional_forward_args=(attention_mask, model), n_steps=1)
            
        # Normalize attributions for better visualization
        attributions = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
        attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min() + 1e-6)
        # Get the tokens and attributions only within the test_code range
        filtered_tokens = tokenized_prompt[test_start_index:test_end_index]
        filtered_tokens = [clean_token(token) for token in filtered_tokens]  # Clean tokens
        filtered_attributions = attributions[test_start_index:test_end_index]

        filtered_tokens_attributions = [(token, attribution) for token, attribution in zip(filtered_tokens, filtered_attributions) if token not in stopwords]

        # Sort tokens based on attributions
        #sorted_tokens = sorted(zip(filtered_tokens_attributions, attributions), key=lambda x: x[1], reverse=True)
        sorted_tokens = sorted(filtered_tokens_attributions, key=lambda x: x[1], reverse=True)
        # Free up memory
        del attributions
        torch.cuda.empty_cache()
    else:
        sorted_tokens = []
    return sorted_tokens 



#====================================================
#====================================================
def init_setup(technique, data_name):
    # specify GPU
    device = torch.device("cuda")
    #device = torch.device("cpu")
    ml_technique=technique.split("-")[0]
    #input_data, target_data, output_layer, dataset_category = preprocess_data(dataset_path, technique)
    dataset_category = "FlakyLens"
    output_layer = 6 # FlakyLens

    where_data_comes = data_name.split("-")[0] 
    os.makedirs(f"{where_data_comes}-result", exist_ok=True)
    return device, ml_technique, dataset_category, output_layer, where_data_comes

def extract_test_method_names(csv_file, output_file):
    """
    Extracts test method names from Java test methods in a CSV file.
    
    Args:
    - csv_file (str): Path to the input CSV file.
    - output_file (str): Path to save the extracted method names.
    """

    # Load CSV
    df = pd.read_csv(csv_file)

    # Regex pattern to extract method names after `@Test public void `
    #method_pattern = re.compile(r'@Test\s+void\s+(\w+)\s*\(')
    method_pattern = re.compile(r'(?:@Test\s+)?(?:public\s+)?void\s+(\w+)\s*\(')

    # Extract method names
    df["test_method_name"] = df["full_code"].apply(lambda code: method_pattern.findall(code)[0] if method_pattern.findall(code) else None)

    # Save extracted method names
    df[["test_method_name"]].to_csv(output_file, index=False)

    print(f"Extracted test method names saved to: {output_file}")
    print(df[["test_method_name"]].head())  # Preview the extracted method names

def parse_category_and_token_list(output_category):
    """Parses the model output to extract the category and tokens."""

    # Regular expressions to extract category and tokens
    category_match = re.search(r"Category:\s*(.+)", output_category)
    tokens_match = re.search(r"Tokens:\s*(\[[^\]]+\])", output_category)

    category = category_match.group(1).strip() if category_match else "Unknown"
    tokens = tokens_match.group(1).strip() if tokens_match else "[]"

    # Convert tokens string to a Python list
    try:
        tokens_list = json.loads(tokens)
    except json.JSONDecodeError:
        tokens_list = []

    return category #, tokens_list

def parse_cr(cr, ml_technique, fold):
    total_weighted_avg_scores=[0, 0, 0]
    total_support=0
    weighted_avg_arrays_list=[]
    category_dict={}

    lines = cr.strip().split('\n')
    # parse the class names and metrics
    classes = []
    metrics = []
    for line in lines[2:-4]:  # skip the first 2 and last 3 lines
        t = line.strip().split()
        classes.append(t[0])
        key=t[0]
        values=[float(x) for x in t[1:]]

        with open("../results/per_Category_Evaluation_"+ml_technique+".txt", "a") as file:
            file.write(fold+":"+key+":" + str(values))
            file.write("\n")

        metrics.append(values)
        if key in category_dict:
            existing_values=category_dict[key]
            updated_values=[existing_values[k] + (values[k]*values[-1]) for k in range(len(values)-1)]
            updated_values.append(existing_values[-1] + values[-1]) #This is for adding support
            category_dict[key] = updated_values
        else:
            initial_val = [(values[i]*values[-1]) for i in range(len(values)-1)]
            initial_val.append(values[-1])
            category_dict[key] = initial_val

    
    #print('metrics=',metrics)
    third_last_line = lines[-3].strip().split()

    accuracy = [float(x) for x in third_last_line[1:]]

    second_last_line = lines[-2].strip().split()
    macro_avg = [float(x) for x in second_last_line[2:]]

    # parse the overall scores
    last_line = lines[-1].strip().split()
    weighted_avg = [float(x) for x in last_line[2:]]
    # print the results
    #print('Classes:', classes)

    total_weighted_avg_scores =  [ total_weighted_avg_scores[idx] + (weighted_avg[idx] * weighted_avg[-1]) for idx in range(3)] 
    total_support +=weighted_avg[-1]

    with open("../results/weighted_avg_for_cv_"+ml_technique+".txt", "a") as file: # Once I get the result, need to divide by 10
        file.write(fold+",")
        file.write(str(weighted_avg))
        file.write("\n")
    return category_dict

def get_evaluation_scores(tn, fp, fn, tp):
    print("get_score method is defined")
    if(tp == 0):
        accuracy = (tp+tn)/(tn+fp+fn+tp)
        Precision = 0
        Recall = 0
        F1 = 0
    else:
        accuracy = (tp+tn)/(tn+fp+fn+tp)
        Precision = tp/(tp+fp)
        Recall = tp/(tp+fn)
        F1 = 2*((Precision*Recall)/(Precision+Recall))
    return accuracy, F1, Precision, Recall

# sett seed for data_loaders for output reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed_value=42):
    """Sets seed for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    torch.cuda.manual_seed_all(seed_value) # if you are using CUDA

def setup_logging():
    """Sets up the logging format and level."""
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO)


# setting the seed for reproducibility, same seed is set to ensure the reproducibility of the result
'''def set_deterministic(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True '''

# train the model
def train(model, train_dataloader, cross_entropy, device, optimizer):
    model.train()
    #total_loss, total_accuracy = 0, 0
    # empty list to save model predictions
    #total_preds = []

    # iterate over batches
    for step, batch in enumerate(train_dataloader):

        # push the batch to gpu
        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch

        # clear previously calculated gradients
        model.zero_grad()
        preds = model(sent_id, mask)
        loss = cross_entropy(preds, labels)
        # add on to the total loss
        #total_loss = total_loss + loss.item()
        # backward pass to calculate the gradients
        loss.backward()
        # progress update after every 50 batches.
        #if step % 50 == 0 and not step == 0:
        #    print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
        #    print('loss=',loss.item())
        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        # update parameters
        optimizer.step()
        # model predictions are stored on GPU. So, push it to CPU
        #preds = preds.detach().cpu().numpy()
        # append the model predictions
        #total_preds.append(preds)
    # compute the training loss of the epoch
    #avg_loss = total_loss / len(train_dataloader)
    # reshape the predictions in form of (number of samples, no. of classes)
    #total_preds = np.concatenate(total_preds, axis=0)
    # returns the loss and predictions
    #return avg_loss, total_preds

def evaluate(model, val_dataloader, cross_entropy, device):

    #print("\nEvaluating..")
    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []
    total_labels = []

    # iterate over batches
    for step, batch in enumerate(val_dataloader):

        # Progress update every 50 batches.
        '''if step % 50 == 0 and not step == 0:

            # Calculate elapsed time in minutes.
            # elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(
                step, len(val_dataloader)))'''

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():

            # model predictions
            preds = model(sent_id, mask)

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds, labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            total_preds.append(preds)
            total_labels.append(labels)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)
    total_labels = np.concatenate(total_labels, axis=0)

    return avg_loss, total_preds, total_labels

def codebert_model_define():	
    model_name = "microsoft/codebert-base"
    model_config = AutoConfig.from_pretrained(model_name, return_dict=False, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    auto_model = AutoModel.from_pretrained(model_name, config=model_config)	
    return model_name, tokenizer, auto_model

def codet5_model_define():
    model_name = "Salesforce/codet5-small"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    auto_model = T5ForConditionalGeneration.from_pretrained(model_name)

    return model_name, tokenizer, auto_model


def t5_small_model_define():	
    model_name = "t5-small"
    #try with codet5 instead of t5
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    auto_model = T5EncoderModel.from_pretrained(model_name)
    model_config = T5ForConditionalGeneration.from_pretrained(model_name).config

    return model_name, tokenizer, auto_model


def gemma2b_model_define():	
    model_name = "google/gemma-2b-it"
    auto_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2b-it",
        #device_map="auto",
        torch_dtype=torch.bfloat16
    ).cuda()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #auto_model = AutoModelForCausalLM.from_pretrained(model_name)
    return model_name, tokenizer, auto_model

def gemma7b_model_define():	
    model_name = "google/gemma-7b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #auto_model = AutoModelForCausalLM.from_pretrained(model_name)
    auto_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        #device_map="cuda",
        torch_dtype=torch.bfloat16
    ).cuda()

    return model_name, tokenizer, auto_model

def codegemma7b_model_define():	
    model_name = "google/codegemma-7b-it"
    auto_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model_name, tokenizer, auto_model

#def llama3_8b_model_define():	
#    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
#    auto_model = AutoModelForCausalLM.from_pretrained(
#        model_name,
#        torch_dtype=torch.bfloat16,
#        low_cpu_mem_usage=True
#    ).cuda()
#    tokenizer = AutoTokenizer.from_pretrained(model_name)
#    return model_name, tokenizer, auto_model

def codellama_7b_instruct_model_define():
    model_name = "codellama/CodeLlama-13b-Instruct-hf"
    auto_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
        #device_map="auto",
    ).cuda()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model_name, tokenizer, auto_model

def mistral_model_define():
    model_name = "mistralai/Ministral-8B-Instruct-2410"

    auto_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        use_auth_token=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model_name, tokenizer, auto_model

def qwen_model_define():
    model_name = "Qwen/Qwen2-7B-Instruct"

    auto_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        #trust_remote_code=True
        use_auth_token=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model_name, tokenizer, auto_model

def deep_seek_coder_model_define():	
    #model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
    #model_name = "deepseek-ai/deepseek-llm-67b-chat"
    #model_name = "deepseek-ai/deepseek-coder-33b-base"
    #model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    #model_name = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
    #model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
    model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    auto_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
        ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model_name, tokenizer, auto_model

def llama3_8b_model_define():	
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    auto_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model_name, tokenizer, auto_model

def llama3_70b_model_define():	
    model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
    auto_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
        #low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model_name, tokenizer, auto_model


def llama_7b_model_define():	
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    #llama-3b-instruct (json form; https://github.com/1rgs/jsonformer); Will ask to generate the json of the category. We can add a little bit descriptions of each category, and the flaky test in general.
    auto_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    #tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side='max_length', truncation=True, pad_token_id=50256)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, truncation=True)
    tokenizer.pad_token_id = tokenizer.encode(tokenizer.pad_token)[0]
    return model_name, tokenizer, auto_model


def fmt_gpu_mem_info(gpu_id=0, brief=True) -> str:
    import torch.cuda.memory

    if torch.cuda.is_available():
        report = ""
        t = torch.cuda.get_device_properties(gpu_id).total_memory
        c = torch.cuda.memory.memory_reserved(gpu_id)
        a = torch.cuda.memory_allocated(gpu_id)
        f = t - a

        report += f"[Allocated {a} | Free {f} | Cached {c} | Total {t}]\n"
        if not brief:
            report += torch.cuda.memory_summary(device=gpu_id, abbreviated=True)
        return report
    else:
        return f"CUDA not available, using CPU"
