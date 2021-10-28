# -*- coding: utf-8 -*-

# Author Yasir Hussain (yaxirhuxxain@yahoo.com)


from .baseTokenizer import tokenizer

class tokenizerSimplifier():
    def __init__(self, language, grammer):
        
        if type(grammer) != list:
            raise Exception("Language Grammer should be of type list")
        
        
        global KEYWORDS
        self.language = language
        KEYWORDS = grammer
        self.Keyword = grammer


        self.Separator = ['(', ')', '{', '}', '[', ']', ';', ',', '.']
        self.Operator = ['>>>=', '>>=', '<<=',  '%=', '^=', '|=', '&=', '/=',
                            '*=', '-=', '+=', '<<', '--', '++', '||', '&&', '!=',
                            '>=', '<=', '==', '%', '^', '|', '&', '/', '*', '-',
                            '+', ':', '?', '~', '!', '<', '>', '=', '...', '->', '::']
                    
        self.Literal_Types = ["Integer","DecimalInteger", "OctalInteger","BinaryInteger","HexInteger","FloatingPoint","DecimalFloatingPoint","HexFloatingPoint","Boolean","Character","String","Null"]
        
    
    
    def lex_to_list_type(self, code=str):
        list_code = []
        
        tokens_list = list(tokenizer(data=code, language=self.language, grammer=KEYWORDS, ignore_errors=False))
        for token in tokens_list:
            list_code.append([str(token).split()[0],token.value])
        
        return list_code

    
    def get_clean_code(self, code=str,):
        
        list_code = self.lex_to_list_type(code)
        
        out_code= ""
        for count, token in enumerate(list_code):
            out_code += " " + token[1]
            
        return out_code
    

    
    def get_simplified_code(self, code=str):
        list_code = self.lex_to_list_type(code)
        
        out_code= ""
        for count,token in enumerate(list_code):
            if token[0] in self.Literal_Types:
                if token[0] in ["Integer","DecimalInteger", "OctalInteger","BinaryInteger","HexInteger"]:
                    out_code += " IntLiteral"
                elif token[0] in ["FloatingPoint","DecimalFloatingPoint","HexFloatingPoint"]:
                    out_code += " FloatLiteral"
                elif token[0] == "Boolean":
                    out_code += " BooleanLiteral"
                elif token[0] == "String":
                    out_code += " StringLiteral"
                elif token[0] == "Null":
                    out_code += " NullLiteral"
                else:
                    out_code += " " +token[0] + "Literal"
            else:
                out_code += " " + token[1]
                
        return out_code
            

def scanner(language, grammer):
    if type(grammer) != list:
        raise Exception("Language Grammer should be of type list")
    
    
    if language not in ('C#', 'Java'):
        raise Exception("Language Grammer should be of type list")
    
    return tokenizerSimplifier(language, grammer)




