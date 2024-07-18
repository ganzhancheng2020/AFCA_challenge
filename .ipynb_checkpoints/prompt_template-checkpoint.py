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
回答规则："""回答用户问题时，严格依据产品名称和保险条款中的内容提供精确且清晰的回答。"""
回答:\
'''
# 仅使用产品名称及保险条款中的语句回答问题，并保持精炼。
# 回答用户问题时，严格依据产品名称和保险条款中内容提供精确且清晰的回答。如果保险条款中未能找到相关的解答，回答段落内容中没有具体信息。
    return qa_prompt.format(name,clause,query)

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