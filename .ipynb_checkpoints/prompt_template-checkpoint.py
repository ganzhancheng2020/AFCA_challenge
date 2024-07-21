def get_sys_prompt():
    sys_prompt = '''你是一个基于保险条款的问答系统，对用户提出的有关保险条款的问题给予准确、清晰的回答。'''
    return sys_prompt

def get_input_template(name,clause,query):
    input_template = '''\
产品名称："""{}"""
保险条款："""{}"""
用户问题："""{}"""
回答:\
'''
    return input_template.format(name,clause,query)


def get_qa_prompt(name,clause,query):
    qa_prompt = '''\
产品名称："""{}"""
保险条款："""{}"""
用户问题："""{}"""
回答规则："""回答用户问题时，严格依据产品名称和保险条款中的内容，给予准确无误且条理清晰的回答，确保回答中涵盖所有至关重要的信息点。"""
回答:\
'''
# Rephrase and expand the question, and Respond.
# 回答用户问题时，严格依据产品名称和保险条款中的内容，给予准确无误且条理清晰的回答，确保回答中涵盖所有至关重要的信息点。
# 回答用户问题时，严格依据产品名称和保险条款中的内容提供精确且清晰的回答。如果在保险条款中找不到答案，告诉用户"段落内容中并未提供具体答案"，并描述保险条裤中所提及的内容。
# 仅使用产品名称及保险条款中的语句回答问题，并保持精炼。
# 回答用户问题时，严格依据产品名称和保险条款中内容提供精确且清晰的回答。如果保险条款中未能找到相关的解答，回答段落内容中没有具体信息。
    return qa_prompt.format(name,clause,query)


def re_query(query):
    re_query_prompt = '''\
用户问题："""{}"""
回答规则："""重新表述并扩展用户问题，确保问题表述清晰且全面"""
回答:\
'''
    return re_query_prompt.format(query)


def get_refine_prompt(query, answer, clause, name):
    refine_prompt = '''\
用户问题："""{}"""
现有答案："""{}"""
产品名称："""{}"""
保险条款："""{}"""
回答规则："""现在，我们有机会根据保险条款来优化这个现有答案（仅在必要时进行）。
根据保险条款中的信息，对现有答案进行改进，以更精确地回答用户问题，确保回答中涵盖所有至关重要的信息。如果保险条款没有帮助，则直接返回现有答案。"""
优化后的答案：\
'''
    return refine_prompt.format(query, answer, clause, name)

qa_prompt1 = '''从保险条款
==========
{}
==========
中找问题
==========
{}
==========
的答案，找到答案就仅使用保险条款中的语句回答问题，找不到答案就用自身知识回答并且告诉用户该信息不是来自文档。
不要复述问题，不要回答无关的内容, 直接开始回答问题。
'''