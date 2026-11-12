"""Conservative deterministic executor. Rules use questions and predicted tables only."""
from decimal import Decimal, InvalidOperation
import re

RULE_VERSION = "1.0"


def number(value):
    text = str(value).strip().replace("\u2212", "-").replace(",", "")
    if text.endswith("%"):
        text = text[:-1]  # percentage points, not fractions, in table arithmetic
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        return None
    try:
        result = Decimal(text)
        return result if result.is_finite() else None
    except InvalidOperation:
        return None


def format_number(value):
    return format(value.normalize(), "f") if value else "0"


def parse_table(text):
    lines = [line.strip() for line in text.replace("<0x0A>", "\n").splitlines() if "|" in line]
    rows = [[cell.strip() for cell in line.split("|")] for line in lines]
    if rows and rows[0][0].casefold() == "title":
        rows = rows[1:]
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError("No rectangular table with a header and data row")
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Ragged predicted table")
    if len(set(cell.casefold() for cell in rows[0])) != len(rows[0]):
        raise ValueError("Duplicate columns")
    labels = [row[0].casefold() for row in rows[1:]]
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate row labels")
    return rows


def intent(question):
    q = question.casefold().strip().rstrip("?")
    if re.search(r"\b(difference|how much more|how much less)\b", q):
        return "difference"
    if re.search(r"\b(higher|greater|larger|lower|smaller)\b", q) and "which" in q:
        return "comparison"
    if re.search(r"\b(maximum|highest|largest|most|max)\b", q):
        return "maximum"
    if re.search(r"\b(minimum|lowest|smallest|least|min)\b", q):
        return "minimum"
    if q.startswith("how many"):
        return "count"
    if re.search(r"\b(what|how much)\b", q) and re.search(r"\b(value|percentage|number|amount)\b", q):
        return "lookup"
    return None


def mentioned(label, question):
    return bool(label.strip()) and re.search(r"(?<!\w)" + re.escape(label.casefold().strip()) + r"(?!\w)", question.casefold()) is not None


def execute(question, table):
    operation = intent(question)
    result = {"operation": operation, "rule_version": RULE_VERSION, "answer": None, "status": "unsupported", "cells": []}
    if operation is None:
        result["reason"] = "No supported operation"
        return result
    if operation != "count" and re.search(r"\b(above|below|greater than|less than|at least|at most|except|excluding|before|after)\b", question.casefold()):
        return {**result, "reason": "Conditional operands unsupported"}
    if operation == "count" and (len(re.findall(r"\b(above|below|greater than|less than)\b", question.casefold())) > 1 or re.search(r"\b(at least|at most|exactly|between|or equal)\b", question.casefold())):
        return {**result, "reason": "Compound or inclusive count conditions unsupported"}
    if len(table) < 2 or len(table[0]) < 2 or any(len(r) != len(table[0]) for r in table):
        return {**result, "status": "invalid_table", "reason": "Nonrectangular table"}
    # Color/position semantics cannot be recovered reliably from textual table headers.
    if re.search(r"\b(red|blue|green|orange|yellow|purple|leftmost|rightmost|median|ratio|average|sum|total|trend|increase|decrease)\b", question.casefold()):
        return {**result, "reason": "Requires unsupported visual or compound semantics"}
    columns = [i for i, name in enumerate(table[0][1:], 1) if mentioned(name, question)]
    if not columns and len(table[0]) == 2:
        columns = [1]
    if len(columns) != 1:
        return {**result, "reason": "Series not uniquely identified"}
    column = columns[0]
    data = [(i, row[0], number(row[column])) for i, row in enumerate(table[1:], 1)]
    if any(value is None for _, _, value in data):
        return {**result, "reason": "Missing or nonnumeric cells"}
    matched = [(i, label, value) for i, label, value in data if mentioned(label, question)]
    chosen, answer = [], None
    if operation == "lookup" and len(matched) == 1:
        chosen = matched
        answer = format_number(chosen[0][2])
    elif operation in ("difference", "comparison") and len(matched) == 2:
        chosen = matched
        if operation == "difference":
            answer = format_number(abs(chosen[0][2] - chosen[1][2]))
        else:
            lower = bool(re.search(r"\b(lower|smaller)\b", question.casefold()))
            if chosen[0][2] != chosen[1][2]:
                answer = (min if lower else max)(chosen, key=lambda x: x[2])[1]
    elif operation in ("maximum", "minimum") and not matched:
        # Only unqualified extrema: conditions would change the candidate set.
        if re.search(r"\b(above|below|between|before|after|except|excluding|than|from|to)\b", question.casefold()):
            return {**result, "reason": "Conditional extremum unsupported"}
        target = (max if operation == "maximum" else min)(value for _, _, value in data)
        chosen = [item for item in data if item[2] == target]
        if len(chosen) == 1:
            label_question = bool(re.search(r"\b(which|who)\b", question.casefold()))
            answer = chosen[0][1] if label_question else format_number(target)
    elif operation == "count":
        threshold = re.search(r"\b(above|below|greater than|less than)\s+([-+]?\d+(?:\.\d+)?%?)", question.casefold())
        if threshold:
            limit = number(threshold[2])
            chosen = [item for item in data if (item[2] > limit if threshold[1] in ("above", "greater than") else item[2] < limit)]
            answer = str(len(chosen))
        elif re.search(r"\b(data points|categories|bars|rows)\b", question.casefold()) and not matched:
            chosen, answer = data, str(len(data))
    if answer is None:
        return {**result, "reason": "Ambiguous operands, ties, or unsupported form"}
    return {**result, "answer": answer, "status": "ok", "cells": [[i, column] for i, _, _ in chosen],
            "values": [format_number(value) for _, _, value in chosen]}
