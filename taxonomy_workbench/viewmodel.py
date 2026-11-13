"""Human-readable explanations derived from committed algorithm records."""
from pathlib import Path
from event_taxonomy.induction.storage import read_json

PHASES = {'enter':'准备检查', 'diagnose':'A · 诊断结构', 'propose':'B · 归纳概念',
          'validate':'C · 验证候选', 'descend':'进入子节点'}
REASONS = {
    'llm_no_induction':'当前层级无需增加中间概念。',
    'llm_accept_and_stop':'当前层级已通过验证，继续检查子节点内部。',
    'candidate_attempts_exhausted':'本轮候选均未通过验证；停止当前层，但继续检查子节点。',
    'maximum_parent_iterations':'当前父节点已达到迭代上限；继续检查子节点。',
    'maximum_new_nodes':'达到本次新增概念的数量上限。',
    'no_candidate_groups':'未找到可用的候选分组，保留当前结构。',
    'repeated_local_structure':'遇到重复结构，停止当前层以避免循环。',
    'maximum_context_visits':'共享节点的检查次数已达到上限。',
    'low_diagnosis_confidence':'诊断置信度不足，未进行归纳。',
    'low_validation_confidence':'验证置信度不足，候选未被接受。',
    'repeated_rejected_candidate':'这个候选之前已经尝试过，不重复验证。',
    'llm_rejected_candidate':'验证未通过，保留修改前的结构，并尝试其他候选。',
}


def translated_reason(reason):
    if reason.startswith('BudgetExceeded:'):return '模型调用已达到预算上限；继续遍历剩余节点。'
    if reason.startswith('ContextLimit:'):return '当前上下文超过配置上限，保留现有结构。'
    if reason.startswith('ContractError:'):return '模型返回格式未通过检查，保留现有结构。'
    return REASONS.get(reason,reason)


def explain(change, nodes):
    action=change['action']; parent=change.get('parent_id','')
    cursor=change.get('cursor',{}); validation=change.get('validation') or change.get('feedback',{}).get('validation') or {}
    added=(change.get('changes') or {}).get('added_nodes',[])
    base={'sequence':change['sequence'],'action':action,'parent_id':parent,
          'parent_label':nodes.get(parent,{}).get('concept_label',parent),
          'round':cursor.get('round',0),'attempt':cursor.get('attempt',0),
          'decision':change.get('decision'),'added_labels':[n['concept_label'] for n in added],
          'llm_ref':change.get('llm_ref'), 'confidence':validation.get('confidence'),
          'reasoning':validation.get('reasoning'),'tone':'neutral','stage':'A'}
    if action=='diagnose_and_group':
        diag=change['diagnosis'];base.update(title='发现可归纳的子群',
            summary='诊断发现 %d 个候选子群。接下来尝试为它们生成合适的中间概念。'%len(change['groups']),
            reasoning=diag['reasoning'],confidence=diag['confidence'],groups=change['groups'])
    elif action=='candidate_constructed':
        base.update(title='生成候选中间层',stage='B',tone='proposal',
                    summary='提出 %d 个中间概念，尚未写入正式结构；下一步验证每条新增上下位关系。'%len(added))
    elif action=='reject_candidate':
        base.update(title='拒绝本次修改',stage='C',tone='rejected',decision='REJECT',
                    summary=translated_reason(change['reason'])+' 这不表示原结构已经最优。')
    elif action=='accept_candidate':
        again=change['decision']=='ACCEPT_AND_CONTINUE'
        base.update(title='接受，继续检查当前层' if again else '接受，当前层稳定',stage='C',tone='accepted',
                    summary=('已新增 %d 个中间概念。'%len(added))+('先重新检查当前父节点的新子节点，还不向下递归。' if again else '本层检查结束，再进入各子节点内部。'))
    elif action=='stop_level':
        base.update(title='本层无需归纳' if change.get('semantically_stable') else '停止当前层',
                    summary=translated_reason(change.get('reason','')),reasoning=change.get('diagnosis',{}).get('reasoning'))
    return base


def timeline(directory,state):
    nodes={n['concept_id']:n for n in state['taxonomy']['nodes']}
    result=[]
    for number in range(1,state['sequence']+1):
        change=read_json(directory/'steps'/('%06d'%number)/'change.json')
        if change['action'] not in ('diagnose_and_group','candidate_constructed','reject_candidate','accept_candidate','stop_level'):continue
        if change.get('reason')=='fewer_than_two_children':continue
        result.append(explain(change,nodes))
    return result


def readiness_error(message):
    if 'input concepts lack a root path' in message or 'definitions are missing' in message:
        return '需要先补齐概念定义，并将细粒度概念关联到高层类别。'
    if 'semantic review is not frozen' in message:return '概念及初始归属尚未完成审核。'
    return '数据未通过完整性检查：'+message
