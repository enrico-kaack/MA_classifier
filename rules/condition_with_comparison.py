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
                
                if self._check_if_direct_comparison(testNode):
                    problems.append({"type": "CONDITION_COMPARISON", "line_number":  testNode.lineno, "col_offset": testNode.col_offset})
        return problems

    def _check_if_direct_comparison(self, node):
        if isinstance(node, ast.BoolOp):
            violated = False
            for n in node.values:
                print("nodes", n)
                if self._check_if_direct_comparison(n):
                    violated = True
            return violated
        if isinstance(node, ast.UnaryOp):
            return self._check_if_direct_comparison(node.operand)
        if not  isinstance(node, ast.Call):
            return True