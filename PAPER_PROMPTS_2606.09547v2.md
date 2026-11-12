# Streaming Interventions 论文提示词整理

来源：当前目录 `2606.09547v2.pdf`，论文 *Streaming Interventions: Can Video Large Language Models Correct Mistakes as They Occur?*，arXiv:2606.09547v2，2026-06-17。以下页码均为 PDF 页脚页码（与文件页序一致）。

本文依据本地 PDF 整理。英文代码块为论文提示词框中的文本，合并了跨页内容，去除了排版换行、行末断词和多余空格；保留原有措辞、占位符及明显笔误。框标题是用途名称，不应自动作为模型输入的一部分。中文说明、流程梳理和 README 对接说明属于整理说明，不是论文原文。

## 1. 评测调用协议

依据正文第 5 节（第 7 页）和附录 D/E（第 14-18 页）：

1. Turn-based 模型每隔 **5 秒**调用一次。
2. 先用完成检测 prompt 判断当前 `[recipe_step]` 是否完成；返回 `YES` 时要求解释完成依据。论文报告该解释要求能提高表现。
3. 若返回未完成，再调用错误检测 prompt，判断即将发生或已经发生的错误。
4. 错误检测返回以 `YES` 或 `NO` 开头；`YES` 后附简短、直接面向用户的纠错反馈。不要把尚未完成本身视为错误。
5. 详细错误检测 prompt 传入最近 **1 分钟**的反馈 `[prev_detected_mistakes]`，避免重复反馈。
6. Per-recipe-step：每个步骤提供对应指令，视频从该步骤真实开始时间重新开始。Full-recipe：视频连续播放（正文第 5 节）。
7. Streaming narration：先收集叙述到 `[narration_list]`，再每 5 秒调用 helper LLM 判断完成情况；未完成时用 D 的简洁错误检测逻辑，以叙述替代视频输入。LiveCC 每秒生成叙述。

### 模型与提示词选择

| 用途 | 模型 | 论文位置 |
|---|---|---|
| 完成检测 | 附录 D 所列 turn-based 模型 | D，第 15 页 |
| 详细错误检测 | Qwen3-VL-Instruct、Qwen3.5、Gemini-Flash 系列 | D，第 16-17 页 |
| 简洁错误检测 | VideoLLaMA3-7B、InternVL3.5-38B、Qwen2.5-VL-Instruct 系列 | D，第 17 页 |
| 叙述完成检测 | Streaming narration 的 helper LLM | E，第 18 页 |
| 反事实指令和反馈生成 | Ego-CoMist 数据生成流程；正文使用 Gemini-2.5-Pro | I，第 20-25 页 |
| 反事实反馈时间推断 | Ego-CoMist 数据生成流程；正文使用 Gemini-2.5-Pro | J，第 26-28 页 |

## 2. 与官方 README 输出格式的关系

README 的 `video_id`、`pred_texts`、`pred_timestamps` 是最终预测文件格式，并不意味着模型每次必须直接返回整个 JSON。论文附录 D 的模型响应格式是 `YES/NO`，错误检测的肯定响应附带反馈。

工程对接时应分别处理模型响应和预测文件序列化：步骤指令以 `Instruction:` 存储，错误反馈以 `Feedback:` 存储，成功确认以 `Success:` 存储，并记录对应视频时间；最终文件是按视频组织的 JSON 列表。这里是结合 README 的对接说明，不是论文另行提供的 JSON prompt，也不是作者公开的完整解析实现。

原先使用的 `Current recipe step`、`Video observation time`、`ALL requirements`、`Do not assume unseen actions occurred`、`completed/reason` JSON 等组合并非附录 D 原文。严格复现时不应将这些自行增加的要求混入下列原始模板。

## 3. 附录 D：Turn-based 评测提示词

以下依次为完成检测、详细错误检测、简洁错误检测。替换 `[recipe_step]`；详细版还需替换 `[prev_detected_mistakes]`。保留模型对应的版本。

### D-1. Check if Recipe Step Complete

来源：附录 D，第 15 页。

```text
You are an expert cooking assistant who is observing a person cook.

##INSTRUCTIONS:

The person is currently at the following recipe step: [recipe_step]. Has the person already completed the recipe step? If the person has completed the recipe step answer ‘YES’ else answer ‘NO’. If you answer ‘YES’ describe why you think the person already completed the recipe step.
```

### D-2. Check for Mistake

来源：附录 D，第 16、17 页。

```text
You are an expert cooking assistant who is observing a person cook. You should look out for mistakes made by the person.

##INSTRUCTIONS: The person is trying to complete the following recipe step: [recipe_step]. Your task is to check if the person is about to make or has already made a mistake. Mistakes occur when the person performs actions that deviates from the instruction and DIRECTLY INTERFERES WITH SUCCESSFUL INSTRUCTION COMPLETION. Do not penalize actions that do not directly interfere with instruction completion (e.g. washing broccoli before cutting, if the instruction just says ‘cut broccoli’).

Here are some common types of mistakes that you should look out for: 1. Technique Error: A mistake in how a step is physically performed. Examples include chopping with the wrong motion, stirring when folding is required, or spilling during transfer, producing uneven cuts or texture issues even when tools and amounts are right. Ignore minor technique errors such as not holding or gripping objects properly, holding with a risk of dropping etc. that do not interfere directly with recipe completion. 2. Preparation Error: A setup mistake before executing the step . Using the wrong or dirty utensil, not washing/peeling/draining ingredients, insufficient draining of fluid, cutting/chopping without peeling which makes correct execution difficult or unsafe. 3. Measurement Error: An error in quantity — wrong counts, volumes, weights, or units. Mixing up teaspoons and tablespoons, misreading a scale, or miscounting items leads to off ratios and predictable taste or texture problems. 4. Temperature Error: A mistake in heat level or thermal state — the applied temperature, starting temperature, or thermal transition is wrong. Not preheating, using the wrong microwave power, overheating oil, or adding cold liquid when warm is required often causes burning, undercooking, or split emulsions. 5. Timing Error: A mistake in duration – over- or under-doing a step or skipping required rests, proofs, or cooling periods. Overcooking, underblending, or cutting resting time short typically yields incorrect doneness or unstable textures.

Do not repeat previously detected mistakes. Here are the feedbacks corresponding to previously detected mistakes:[prev_detected_mistakes]. Assume the recipe step is still in progress. Your task is to identify any mistake that’s already visible in the partially completed step. Do no penalize partially competed recipe steps. If you observe a mistake answer ‘YES’, else ‘No’. Your response MUST BEGIN WITH ‘YES’ or ‘NO’. In case you answer ‘YES’, please follow with a concise feedback to the user describing the mistake (i.e. YES. <feedback>.). Directly address the person.
```

### D-3. Check for Mistake

来源：附录 D，第 17 页。

```text
You are an expert cooking assistant who is observing a person cook. You should look out for mistakes made by the person.

##INSTRUCTIONS: The person is trying to complete the following recipe step: [recipe_step]. Your task is to check if the person is about to make or has already made a mistake. Mistakes occur when the person performs actions that deviates from the instruction and DIRECTLY INTERFERES WITH SUCCESSFUL INSTRUCTION COMPLETION. Assume the recipe step is still in progress. Your task is to identify any mistake that’s already visible in the partially completed step.

Do no penalize partially competed recipe steps. If you observe a mistake answer ‘YES’, else ‘No’. Your response MUST BEGIN WITH ‘YES’ or ‘NO’. In case you answer ‘YES’, please follow with a concise feedback to the user describing the mistake (i.e. YES. <feedback>.). Directly address the person.
```

## 4. 附录 E：Streaming narration 评测提示词

输入为叙述列表。论文框内未明确给出高层指令的独立占位符位置，不能把自行补充的字段冒充原文。

### E-1. Check if Recipe Step Complete (Streaming Narration Models)

来源：附录 E，第 18 页。

```text
You are an intelligent chatbot that is judging another system which narrates human cooking videos. Given a high level action instruction and a list of narrations generated from the system, your job is to decide if the narration is correct and shows completion of the instruction.

##INSTRUCTIONS: List of narrations: [narration_list]

Answer ‘YES’ if the instruction is completed otherwise output ‘NO’.
```

## 5. 附录 I：Ego-CoMist 第一阶段提示词

用于离线构造反事实指令和反馈，不是在线评测输入。顺序为 I.1 数量（第 20-21 页）、I.2 准备（第 21-22 页）、I.3 技法（第 22-23 页）、I.4 温度（第 23-24 页）、I.5 时长（第 24-25 页）。逐框保留如下；Greater/Higher 等标题仅对应论文实际展示的版本。

### I-1. Extract Attributes from Recipe Step

来源：附录 I，第 20 页。

```text
You are an expert cooking assistant. You are watching a person follow a step by step recipe.

##INSTRUCTIONS: The person is following the recipe step: [org_recipe_step].

Does this recipe step involve measuring a specific quantity or amount of an ingredient?

Examples include specific quantities like teaspoon, teaspoons, tablespoons, liters, ounces, grams, kgs, kilograms, cups, pints, quarts, gallons etc.

Extract the explicitly stated numerical quantity or amount and Answer ‘YES’ or ‘NO’. If ‘YES’, then return the specific quantity and ingredient. Return a dict with two fields: {‘ingredient’: .., ‘quantity’: ...}. DO NOT RETURN ANYTHING ELSE.
```

### I-2. Counterfactual Attributes (Greater)

来源：附录 I，第 20 页。

```text
You are an expert cooking assistant.

##INSTRUCTIONS: The following recipe step: [org_recipe_step]; uses the following quantity: [quantity] of the item: [ingredient]. Suggest an alternative reasonable quantity of the item greater than [quantity] for the recipe step. RETURN JUST THE QUANTITY. DO NOT RETURN ANYTHING ELSE.
```

### I-3. Counterfactual Instruction

来源：附录 I，第 20 页。

```text
You are an expert cooking assistant.

##INSTRUCTIONS: Modify the instruction [org_instruction] for the recipe step: [org_recipe_step]; such that the recipe step uses [counterfactual_quantity] of [ingredient]. RETURN JUST THE INSTRUCTION DO NOT RETURN ANYTHING ELSE.
```

### I-4. Counterfactual Feedback

来源：附录 I，第 21 页。

```text
You are an expert cooking assistant. You are assisting a person step by step through a recipe.

##INSTRUCTIONS: Instead of using the specified quantity: [quantity] of [ingredient], the person instead used the following quantity: [counterfactual_quantity]. Given the correct specified quantity and the incorrectly used quantity provide an appropriate feedback message. The feedback message should be short and one line in length. Make sure to point out the mistake clearly in the feedback message in a single line. RETURN JUST THE FEEDBACK MESSAGE DO NOT RETURN ANYTHING ELSE.
```

### I-5. Extract Attributes and Propose Counterfactual Attributes

来源：附录 I，第 21 页。

```text
You are an expert cooking assistant.

##INSTRUCTIONS: Given a recipe step, your task is to propose variants of that recipe step that use an alternative preparation method. Alternative preparation methods include: different blending/whisking/mixing/beating methods; using different utensils or ingredients; using left instead of right hands and vise versa; dealing with fluids differently; cutting or chopping ingredients differently.

Now, given the following recipe step: [org_recipe_step]. Extract the specific preparation method used, the food item or cooking utensil used (object) and then propose an alternative preparation method. Make sure that the proposed alternative preparation method is realistic and likely to be provided by a expert cooking assistant.

Return a dict with five keys: {‘current_prep_method’: ..., ‘object’: ... , ‘alternative_prep_method’: ... , ‘current_prep_method_duration’: ..., ‘alternative_prep_method_duration’: ...}. Where, if the current preparation method (current_prep_method) requires cooking (e.g. heating, boiling, microwaving) for a specific duration this duration is returned in [current_prep_method_duration] field (None otherwise); and if the alternative preparation method (alternative_prep_method_duration) requires cooking (e.g. heating, boiling, microwaving) for a specific duration this duration is returned in [alternative_prep_method_duration] field (None otherwise). Keep the [alternative_prep_method] in the dict very short. RETURN JUST THE COOKING PREP METHOD DICT DO NOT RETURN ANYTHING ELSE.
```

### I-6. Counterfactual Instruction

来源：附录 I，第 21、22 页。

```text
You are an expert cooking assistant.

##INSTRUCTIONS:

Generate a cooking instruction that uses the following preparation method: [alternative_prep_method] on: [object] for: [alternative_prep_method_duration]. Here is an example of a similar instruction: [org_instruction]. RETURN JUST THE INSTRUCTION DO NOT RETURN ANYTHING ELSE.
```

### I-7. Counterfactual Feedback

来源：附录 I，第 22 页。

```text
You are an expert cooking assistant.

##INSTRUCTIONS: Instead of the correct preparation method: [org_prep_method] the person used the preparation method: [alternative_prep_method] – on the food item or cooking utensil: [object]. Provide an appropriate feedback message to the person. The feedback message should be one line in length and clearly point out the mistake made by the person. Make sure to point out the mistake clearly in the feedback message in a single line. RETURN JUST THE FEEDBACK MESSAGE DO NOT RETURN ANYTHING ELSE.
```

### I-8. Extract Attributes from Recipe Step

来源：附录 I，第 22 页。

```text
You are an expert cooking assistant. You are assisting a person step by step through a recipe.

##INSTRUCTIONS: The person is trying to complete the following recipe step: [org_recipe_step].

If the recipe step uses a specific cooking technique, example techniques include: roll, pat, squeeze, hold, measuring, adding, transferring, chopping, cutting, peeling, spiralizing, flipping, stirring, whisking, beating, return the cooking technique, return a dict with the cooking technique and the food item or ingredient that it is used on: {‘technique’: ..., ‘item’:...}. If not, return None. DO NOT RETURN ANYTHING ELSE.
```

### I-9. Counterfactual Attributes

来源：附录 I，第 22 页。

```text
You are an expert cooking assistant.

##INSTRUCTIONS: Given a recipe step, [org_recipe_step] that uses the cooking technique: [technique] on [item], your task is to propose an alternative cooking technique. Make sure that the proposed alternative cooking technique is realistic given the recipe step. RETURN JUST THE ALTERNATIVE COOKING TECHNIQUE DO NOT RETURN ANYTHING ELSE.
```

### I-10. Counterfactual Instruction

来源：附录 I，第 23 页。

```text
You are an expert cooking assistant.

##INSTRUCTIONS: Given the cooking instruction [org_instruction], generate a new instruction that uses the following cooking technique instead: [alternative_technique]. RETURN JUST THE INSTRUCTION DO NOT RETURN ANYTHING ELSE.
```

### I-11. Counterfactual Feedback

来源：附录 I，第 23 页。

```text
You are an expert cooking assistant.

##INSTRUCTIONS: Instead of the correct technique: [counterfactual_technique] the person used the technique: [technique] – on the: [item]. Provide an appropriate feedback message to the person. The feedback message should be one line in length and clearly point out the mistake made by the person. Make sure to point out the mistake clearly in the feedback message in a single line. RETURN JUST THE FEEDBACK MESSAGE DO NOT RETURN ANYTHING ELSE.
```

### I-12. Extract Attributes from Recipe Step

来源：附录 I，第 23 页。

```text
You are an expert cooking assistant. You are assisting a person step by step through a recipe.

##INSTRUCTIONS: The person is following the recipe step: [org_recipe_step]

Does this recipe step involve cooking an ingredient or a food item on a kitchen appliance or utensil at a specific temperature or heat setting? Examples include setting an oven or a stove at a specific temperature or heat setting. If yes, then return the specific kitchen appliance or utensil (kitchen item), temperature or heat setting (heat setting) and the food item or ingredient (food item). Return a dict with three fields: {‘kitchen_item’: .., ‘heat_setting’: ..., ‘food_item’: ...} else return None. DO NOT RETURN ANYTHING ELSE.
```

### I-13. Counterfactual Attribute (Higher)

来源：附录 I，第 23 页。

```text
You are an expert cooking assistant.

##INSTRUCTIONS: The following recipe step: [org_recipe_step]; uses the following kitchen appliance/utensil: [kitchen_item]; to cook the following food item: [food_item]; at the following heat setting: [heat_setting]. Suggest an alternative reasonable heat setting higher than [heat_setting] for the recipe step. RETURN JUST THE HEAT SETTING IF POSSIBLE ELSE RETURN NONE. DO NOT RETURN ANYTHING ELSE.
```

### I-14. Counterfactual Instruction

来源：附录 I，第 24 页。

```text
You are an expert cooking assistant.

##INSTRUCTIONS: Modify the cooking instruction: [org_instruction]; such that the recipe step uses the following heat setting: [counterfactual_heat_setting] – while cooking [food_item] on the utensil/appliance: [kitchen_item]. RETURN JUST THE INSTRUCTION DO NOT RETURN ANYTHING ELSE.
```

### I-15. Counterfactual Feedback

来源：附录 I，第 24 页。

```text
You are an expert cooking assistant.

##INSTRUCTIONS: Instead of using the correct heat setting: [heat_setting], the person instead used a incorrect heat setting: [counterfactual_heat_setting] – on the food item: [food_item] – and using the kitchen utensil/appliance: [kitchen_item].

Provide an appropriate feedback message. The feedback message should be short and one line in length. Make sure to point out the heat setting clearly in a feedback message in a single line. RETURN JUST THE FEEDBACK MESSAGE DO NOT RETURN ANYTHING ELSE.
```

### I-16. Extract Attributes from Recipe Step

来源：附录 I，第 24 页。

```text
You are an expert cooking assistant. You are assisting a person step by step through a recipe.

##INSTRUCTIONS: The person is following the recipe step: [org_recipe_step]. Does this recipe step involve cooking an ingredient or a food item on a kitchen appliance or utensil for a specific amount of time? Examples include setting an oven or a stove at a specific temperature or heat setting.

If yes, then return the specific kitchen appliance or utensil (kitchen_item), the amount of time (amount_of_time) and the food item or ingredient (food_item). Return a dict with three fields: {‘kitchen_item’: .., ‘amount_of_time’: ..., ‘food_item’: ...}. Else return None. DO NOT RETURN ANYTHING ELSE.
```

### I-17. Counterfactual Attribute (Greater)

来源：附录 I，第 24、25 页。

```text
You are an expert cooking assistant.

##INSTRUCTIONS: The following recipe step: [org_recipe_step]; uses the following kitchen appliance/utensil: [kitchen_item]; to the cook the following food item: [food_item]; for the following amount of time: [amount_of_time]. Suggest an alternative reasonable amount of time significantly greater (at least 15 seconds) than [amount_of_time] for the recipe step. RETURN JUST THE AMOUNT OF TIME IF POSSIBLE ELSE RETURN NONE. DO NOT RETURN ANYTHING ELSE.
```

### I-18. Counterfactual Instruction

来源：附录 I，第 25 页。

```text
You are an expert cooking assistant.

##INSTRUCTIONS: Modify the cooking instruction: [org_instruction]; such that the recipe step uses the following amount of time: [counterfactual_amount_of_time] – while cooking [food_item] on the utensil/appliance: [kitchen_item]. RETURN JUST THE INSTRUCTION DO NOT RETURN ANYTHING ELSE.
```

### I-19. Counterfactual Feedback (Greater)

来源：附录 I，第 25 页。

```text
You are an expert cooking assistant. You are assisting a person step by step through a recipe.

##INSTRUCTIONS: Instead of the using the correct amount of time for cooking: [amount_of_time, the person did not cook the food item: [food_item] – and using the kitchen utensil/appliance: [kitchen_item] – for long enough. Provide an appropriate feedback message. The feedback message should be short and one line in length. Make sure to point out that the person did not cook for long enough clearly in the feedback message in a single line. RETURN JUST THE FEEDBACK MESSAGE DO NOT RETURN ANYTHING ELSE.
```

## 6. 附录 J：Ego-CoMist 第二阶段提示词

用于离线推断反事实反馈时间；模型可见完整步骤叙述和反事实信息，是 oracle 数据生成设置，不应当作在线模型可见输入。正文第 4.2 节说明：缺少叙述时用 Qwen3-VL-32B-Instruct 生成，采用 10 秒窗口、5 秒步长，关注手与物体交互。以下依次为 J.1 数量、J.2 准备、J.3 技法、J.4 温度、J.5 两种时长模板。

### J-1. Feedback Timestamp (Greater Counterfactual Quantity)

来源：附录 J，第 26 页。

```text
You are an expert cooking assistant. You are watching a person follow a step by step recipe.

##INSTRUCTIONS: You have provided the person with the following instruction: [org_instruction].

You are now provided with the following step by step account of the person’s activities, along with the corresponding timestamps: [step_by_step_instruction].

Your task is to find the timestamp at which it becomes apparent that the person uses less than the instructed quality of the following: [ingredient]. The person was instructed to use the following quantity [quantity] but instead used [counterfactual_quantity].

WAIT and make sure that the person does not intend to use more (the correct quantity).

Choose one of the following options: [timestamps]. Return the timestamp in seconds from the options above as a python dict of the form: {‘timestamp’: ...}. DO NOT RETURN ANYTHING OTHER THAN THE PYTHON DICT.
```

### J-2. Feedback Timestamp

来源：附录 J，第 26 页。

```text
You are an expert cooking assistant. You are watching a person follow a step by step recipe.

##INSTRUCTIONS: You have provided the person with the following instruction: [org_instruction].

You are now provided with the following step by step account of the person’s activities, along with the corresponding timestamps: [step_by_step_instruction].

Your task is to find the timestamp at which it becomes apparent that the person has used the following (wrong) preparation method: [current_prep_method] – on the food item or cooking utensil: [object] – instead of the correct preparation method: [alternative_prep_method].

Choose one of the following options: [timestamps]. Return the timestamp in seconds from the options above as a python dict of the form: {‘timestamp’: ...}. DO NOT RETURN ANYTHING OTHER THAN THE PYTHON DICT.
```

### J-3. Feedback Timestamp

来源：附录 J，第 27 页。

```text
You are an expert cooking assistant. You are watching a person follow a step by step recipe.

##INSTRUCTIONS: You have provided the person with the following instruction: [org_instruction].

You are now provided with the following step by step account of the person’s activities, along with the corresponding timestamps: [step_by_step_instruction].

Your task is to find the timestamp at which it becomes apparent that the person is using the wrong technique: [technique] – and the person does not intend to use the correct technique: [alternative_technique].

Choose one of the following options: [timestamps]. Return the timestamp in seconds from the options above as a python dict of the form: {‘timestamp’: ...}. DO NOT RETURN ANYTHING OTHER THAN THE PYTHON DICT.
```

### J-4. Feedback Timestamp

来源：附录 J，第 27 页。

```text
You are an expert cooking assistant. You are watching a person follow a step by step recipe.

##INSTRUCTIONS: You have provided the person with the following instruction: [org_instruction].

You are now provided with the following step by step account of the person’s activities, along with the corresponding timestamps: [step_by_step_instruction].

Your task is to find the timestamp at which it becomes apparent that the person uses the wrong heat/temperature setting: [heat_setting] – of the utensil/appliance: [kitchen_item] – to cook the following: [food_item]– instead of the correct heat/temperature setting: [counterfactual_heat_setting].

Choose the EARLIEST timestamp where it becomes clear that the person has chosen the incorrect heat/temperature setting. (This is usually when the person starts an appliance or adjusts the heat setting).

Choose one of the following options: [timestamps]. Return the timestamp in seconds from the options above as a python dict of the form: {‘timestamp’: ...}. DO NOT RETURN ANYTHING OTHER THAN THE PYTHON DICT.
```

### J-5. Feedback Timestamp (Smaller Counterfactual Duration)

来源：附录 J，第 28 页。

```text
You are an expert cooking assistant. You are watching a person follow a step by step recipe.

##INSTRUCTIONS: You have provided the person with the following instruction: [org_instruction].

You are now provided with the following step by step account of the person’s activities, along with the corresponding timestamps: [step_by_step_instruction].

First find the [start_timestamp] when the person starts to cook: [food_item] – using the utensil/appliance: [kitchen_item]. Now, instead of the specified amount of time: [amount_of_time] – the person cooks for: [counterfactual_amount_of_time]. Find the timestamp [critical_timestamp] at which it becomes apparent that the person does not cook: [food_item] – using the utensil/appliance: [kitchen_item] – for long enough.

Choose one of the following options: [timestamps]. Choose the LAST timestamp where it becomes clear that the person has not cooked for long enough, after the person stops cooking [critical_timestamp] with the [kitchen_item], at least [amount_of_time] from the beginning of the cooking process [start_timestamp].

Return the timestamps in seconds from the options above as a python dict of the form: {‘start_timestamp’: ..., ‘critical_timestamp’: ...}. The ‘start_timestamp’ and ‘critical_timestamp’ should be at least [amount_of_time] seconds apart. If you are not sure of the [start_timestamp] or [critical_timestamp] return None. DO NOT RETURN ANYTHING OTHER THAN THE PYTHON DICT.
```

### J-6. Feedback Timestamp (Larger Counterfactual Duration)

来源：附录 J，第 28 页。

```text
You are an expert cooking assistant. You are watching a person follow a step by step recipe.

##INSTRUCTIONS: You have provided the person with the following instruction: [org_instruction].

You are now provided with the following step by step account of the person’s activities, along with the corresponding timestamps: [step_by_step_instruction].

First find the [start_timestamp] when the person starts to cook: [food_item] – using the utensil/appliance: [kitchen_item]. Now, instead of the specified amount of time: [amount_of_time] – the person cooks for: [counterfactual_amount_of_time]. Find the timestamp [critical_timestamp] at which it becomes apparent that the person cooks: [food_item] – using the utensil/appliance: [kitchen_item] – for too long.

Choose one of the following options: [timestamps]. Choose the LAST timestamp where it becomes clear that the person has cooked for too long, about [amount_of_time] from the beginning of the cooking process [start_timestamp] with the [kitchen_item] (don’t wait too long).

Return the timestamps in seconds from the options above as a python dict of the form: {‘start_timestamp’: ..., ‘critical_timestamp’: ...}. The ‘start_timestamp’ and ‘critical_timestamp’ should be at least [amount_of_time] seconds apart. If you are not sure of the [start_timestamp] or [critical_timestamp] return None. DO NOT RETURN ANYTHING OTHER THAN THE PYTHON DICT.
```

## 7. 占位符及原文疑点

| 占位符 | 用途 |
|---|---|
| `[recipe_step]` | 在线评测的当前步骤 |
| `[prev_detected_mistakes]` | 最近一分钟已生成反馈 |
| `[narration_list]` | Streaming narration 的叙述缓冲 |
| `[org_recipe_step]`、`[org_instruction]` | 数据生成的原始步骤和指令 |
| 数量、准备方法、技法、温度、时长相关属性 | 第一阶段提取或构造，供后续模板填充 |
| `[step_by_step_instruction]` | 第二阶段提示词框内使用的带时间叙述占位符 |
| `[timestamps]` | 第二阶段允许选择的时间戳集合 |
| `[start_timestamp]`、`[critical_timestamp]` | 时长错误的起点及关键反馈时间 |

核对原文时需要保留并注意：

- D 的错误检测中有 `Do no penalize partially competed recipe steps` 等笔误；本文未静默修订。
- E 对 helper 模型名称有 Qwen3.5-27B / Qwen3.3-27B 的不一致；正文第 5 节还出现 Qwen3-32B。实现时应明确选择并记录，不能宣称三者一致。
- E 的完成检测框提到 high level action instruction，但只显式列出 `[narration_list]`。
- I.2 的准备反馈框使用 `[org_prep_method]`，属性提取框使用 `[current_prep_method]`；I.3 在不同框使用 `[alternative_technique]` / `[counterfactual_technique]`。变量需要显式映射。
- I.5 反馈框的 `[amount_of_time` 缺少右括号。本文保留。
- J 正文说明使用 `[step_by_step_descriptions]`，提示词框实际使用 `[step_by_step_instruction]`。
- I/J 若干数量、温度和时长模板里“原始值”和“反事实值”的正确/错误角色，与正文第 4 节、附录 H 的叙述存在不一致风险。例如 J.1 标题是 Greater Counterfactual Quantity，但框内将 instructed quantity 写为 `[quantity]`、实际使用写为 `[counterfactual_quantity]`，同时要求查找“使用量不足”。不要直接假定变量名已经保证语义正确；复现前应逐条对照正文与生成样例。此处仅指出疑点，不替作者改写。
- I.1、I.4、I.5 的说明提到大小/高低等不同方向，但没有逐一展示所有方向的模板。本文只收录 PDF 实际出现的框，不自行补造缺失版本。
- J 的时长模板要求返回 Python dict；这与严格 JSON 不等价。I 中也有 Python 风格的单引号和 `None`。实现解析时应区分，但不要把修改后的格式标成论文原文。

## 8. 复现边界

本文收录 PDF 中 D/E/I/J 的全部提示词框，未另造系统提示词、时间戳前缀或 JSON 约束。论文没有在这些框中完整指定每种模型的消息角色分配、图像打包格式、帧采样参数、完整历史裁剪及成功确认文本映射；这些工程细节不能仅凭 prompt 框断言已完全复现。

附录 F（第 18 页）说明评测时间窗口为 30 秒。这是指标匹配窗口，不是 5 秒的模型调用间隔，也不是叙述生成的 10 秒窗口。
