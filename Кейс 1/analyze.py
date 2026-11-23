import pandas as pd
import numpy as np
from pathlib import Path

def working_hours(a, b):
    if pd.isna(a) or pd.isna(b) or b <= a:
        return 0.0
    days = pd.date_range(a.normalize(), b.normalize(), freq='D')
    total = 0.0
    for d in days:
        if d.weekday() < 5:
            total += max(0.0, (min(b, d + pd.Timedelta(hours=18)) - max(a, d + pd.Timedelta(hours=9))).total_seconds() / 3600)
    return total

def add(pid, kind, why):
    records.append((pid, kind, why))

ROOT = Path(__file__).resolve().parent
x = pd.read_csv(ROOT / 'students.csv', encoding='utf-8-sig').sort_values(['ProcessInstanceId', 'EventIndex'])
x['dt'] = pd.to_datetime(x.Timestamp, utc=True).dt.tz_convert('Europe/Moscow')
created = x[x.EventType == 'task_created'].set_index('TaskId')
closed = x[x.EventType == 'task_completed'].copy()
closed['start'] = closed.TaskId.map(created.dt)
closed['hours'] = (closed.dt - closed.start).dt.total_seconds() / 3600
closed['work_hours'] = [working_hours(a, b) for a, b in zip(closed.start, closed.dt)]
stats = {}
for (typ, stage), z in closed.groupby(['RequestType', 'ActivityName']):
    if len(z) < 30: # Пропускаем группы с менее чем 30 заданиями для надёжности статистики
        continue
    vals = z.work_hours.to_numpy()
    q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    stats[(typ, stage)] = {'n': len(z), 'median': med, 'high': max(q3 + 3 * iqr, med * 4, 2.0)} # Выбираем наибольший из трёх порогов (high), чтобы учитывать разброс длительностей и уменьшить число ложных аномалий
records = []
ROLES = {'Проверка заявки': 'specialist', 'Согласование руководителем': 'manager', 'Доработка заявки': 'requester', 'Согласование бюджета': 'finance', 'Проверка договора': 'legal', 'Исполнение заявки': 'procurement', 'Внеплановая проверка': 'auditor'}
for pid, g in x.groupby('ProcessInstanceId', sort=False):
    typ = g.RequestType.iloc[0]
    finish = g[g.EventType == 'process_completed']
    if len(finish):
        done = finish.Result.iloc[0]
    else:
        done = None
    tasks = closed[closed.ProcessInstanceId == pid].sort_values('EventIndex')
    stages = tasks.ActivityName.tolist()
    manager = tasks[tasks.ActivityName == 'Согласование руководителем']
    accepted = manager[manager.Result == 'accept']
    rejected = manager[manager.Result == 'reject']
    budget = tasks[tasks.ActivityName == 'Согласование бюджета']
    legal = tasks[tasks.ActivityName == 'Проверка договора']
    execution = tasks[tasks.ActivityName == 'Исполнение заявки']
    checks = tasks[tasks.ActivityName == 'Проверка заявки']
    revisions = tasks[tasks.ActivityName == 'Доработка заявки']
    audits = tasks[tasks.ActivityName == 'Внеплановая проверка']
    if done == 'fulfilled':
        for stage in ['Проверка заявки', 'Согласование руководителем', 'Согласование бюджета', 'Исполнение заявки'] + (['Проверка договора'] if typ == 'contract' else []):
            if stage not in stages:
                add(pid, 'Пропущен обязательный этап', f'Заявка {typ} завершена fulfilled, но отсутствует закрытие этапа «{stage}».')
        if not accepted.empty and (not budget.empty) and (budget.EventIndex.min() < accepted.EventIndex.min()):
            add(pid, 'Нарушен порядок этапов', 'Проверка бюджета завершена до положительного согласования руководителя.')
        if not execution.empty and (not budget.empty) and (execution.EventIndex.min() < budget.EventIndex.min()):
            add(pid, 'Нарушен порядок этапов', 'Исполнение заявки завершено до проверки бюджета.')
        if typ == 'contract' and (not legal.empty) and (not execution.empty) and (execution.EventIndex.min() < legal.EventIndex.min()):
            add(pid, 'Нарушен порядок этапов', 'Договорная заявка исполнена до юридической проверки.')
        if not accepted.empty and (not checks.empty) and (accepted.EventIndex.min() < checks.EventIndex.min()):
            add(pid, 'Нарушен порядок этапов', 'Согласование руководителя завершено до проверки заявки.')
    if done == 'rejected' and rejected.empty:
        add(pid, 'Несогласованный отказ', 'Процесс завершён как rejected, но нет отказа руководителя.')
    if done == 'fulfilled' and (not rejected.empty):
        add(pid, 'Противоречивый результат', 'Заявка исполнена, хотя присутствует отказ руководителя.')
    if typ != 'contract' and (not legal.empty):
        add(pid, 'Необычный маршрут', f'Для заявки {typ} выполнена юридическая проверка, предусмотренная для contract; нужна проверка основания.')
    if len(audits):
        add(pid, 'Внеплановая проверка', f'Зафиксировано {len(audits)} внеплановых проверок аудитора; требуется подтверждение основания, редкость сама по себе не нарушение.')
    if len(revisions) >= 4:
        add(pid, 'Многократные возвраты', f'Заявка возвращалась на доработку {len(revisions)} раз; обычные маршруты имеют 0–1 возврат.')
    for _, r in manager[manager.Result == 'return'].iterrows():
        later = tasks[tasks.EventIndex > r.EventIndex]
        if done is None:
            continue
        if later.empty:
            add(pid, 'Нарушение маршрута возврата', f'После возврата руководителя (событие {r.EventIndex}) отсутствует доработка.')
        elif later.ActivityName.iloc[0] != 'Доработка заявки':
            add(pid, 'Нарушение маршрута возврата', f'После возврата руководителя (событие {r.EventIndex}) первым не следует завершение доработки.')
    if len(checks) > len(revisions) + 1 and done is not None:
        add(pid, 'Повторная проверка без возврата', f'Проверок заявки {len(checks)}, доработок {len(revisions)}; повторная проверка не объясняется возвратом.')
    for _, t in tasks.iterrows():
        if pd.isna(t.start):
            add(pid, 'Нарушение целостности журнала', f'Нет события создания задания {t.TaskId}.')
        elif t.hours < 0:
            add(pid, 'Нарушение времени', f'Задание «{t.ActivityName}» завершено раньше создания.')
        expected = ROLES.get(t.ActivityName)
        if expected and t.Role != expected:
            add(pid, 'Неожиданная роль', f'Этап «{t.ActivityName}» закрыт ролью {t.Role} вместо {expected}')
        st = stats.get((typ, t.ActivityName))
        if st and st['n'] >= 30: # Проверяем длительность только для групп с 30 и более заданиями, чтобы статистика была надёжнее
            if t.work_hours > st['high']:
                add(pid, 'Необычно долгое задание', f'Этап «{t.ActivityName}» выполнялся {t.work_hours:.2f} ч при пороге {st["high"]:.2f} ч.')
            if t.hours <= 0.05 and st['median'] >= 0.5:
                add(pid, 'Необычно быстрое задание', f'Этап «{t.ActivityName}» выполнен за {t.hours * 60:.1f} мин при медиане {st["median"]:.2f} ч.')
out = pd.DataFrame(records, columns=['ProcessInstanceId', 'AnomalyType', 'Explanation']).drop_duplicates().sort_values(['ProcessInstanceId', 'AnomalyType'])
out.to_csv(ROOT / 'anomalies.csv', index=False, encoding='utf-8')
