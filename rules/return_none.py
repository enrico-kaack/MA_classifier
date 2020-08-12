import asttokens, ast

class ReturnNoneRule():
    def analyse_source_code(self, code):
        python_ast =  asttokens.ASTTokens(code, parse=True)
        problems =  self.analyse_ast(python_ast.tree)
        return problems

    def analyse_ast(self, a):
        problems = []
        for node in ast.walk(a):
            if isinstance(node, ast.Return):
                return_value = node.value
                if isinstance(return_value, ast.Constant):
                    if return_value.value == None:
                        problems.append({"type": "RETURN_NULL", "line_number":  node.lineno, "col_offset": node.col_offset})
        return problems