import asttokens, ast

class ConditionComparison():
    def analyse_source_code(self, code):
        python_ast =  asttokens.ASTTokens(code, parse=True)
        problems =  self.analyse_ast(python_ast.tree)
        return problems

    def analyse_ast(self, a):
        problems = []
        for node in ast.walk(a):
            #for all if
            if isinstance(node, ast.If):
                testNode = node.test
                if not isinstance(testNode, ast.Call):
                    problems.append({"type": "CONDITION_COMPARISON", "line_number":  testNode.lineno, "col_offset": testNode.col_offset})
        return problems