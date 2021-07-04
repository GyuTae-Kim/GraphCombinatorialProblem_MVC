import numpy as np

from .base_instance import BaseInstance


class MVC_Instance(BaseInstance):

    def __init__(self, n_node, feature):
        super(BaseInstance, self).__init__(n_node, feature)
        '''
        아래 변수들 초기값 설정할 것
        나머지는 부모 클래스에서 처리됨
        self._adj
        self._x
        self._weights
        '''
    
    def move(self, next_node, idx=None):
        '''
        idx는 Help Function에 의해 삽입되는 곳이 달라질 경우 사용
        idx가 None일 경우 마지막 노드에 단순 삽입(append)
        next_node는 last_node에 저장
        '''
        pass
