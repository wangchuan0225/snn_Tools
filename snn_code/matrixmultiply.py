"""
文件类型：算子支持文件
算子： matrix_muti
支持类型：类
"""
from src.generator.prim import *
from src.generator.cycle import *
class MatrixMultiply:
    def __init__(self, node_info, addrT):
        # 输入特征图的形状
        In_Shape = node_info['in_shapes'][0]
        # 矩阵的高
        self.In_H = In_Shape[0]
        # 规定矩阵的宽
        self.In_W = In_Shape[1]
        
        # 权重的形状
        self.Weight_H = node_info['weight_shape'][0]
        self.Weight_W = node_info['weight_shape'][1]
        
        # 权重值
        self.all_weights = node_info['weight']
        
        # 输出特征图的形状
        self.Out_H = self.Weight_H
        self.Out_W = self.In_W
        
        # 初始化一些计算相关的参数
        # 有效的权重行数
        self.valid_sa_h = SA_H
        # 输入数据向量: 当中一个 tstep的大小
        self.whole_tstep_bk = int(self.In_H * TSTEP / 8)
        # 输出数据向量: 当中一个tstep的大小
        self.out_whole_tstep_bk = int(self.Out_H * TSTEP / 8)
        # tstep中一个元素的大小
        self.element_bk = int(T_LEN / 8)
        # 有效权重行在内存中所占字节数
        self.sa_line_byte = int(SA_H / 8)
        # 权重宽所占字节数
        self.weight_aline_bk = int(self.Weight_W * SIZE_OF_WEIGHT)
        # 全局资源对象
        self.addrT = addrT
        
        # 初始化权重和输出空间
        out_size = self.Out_H * self.Out_W * int(T_LEN / 8)
        w_par_size_c = SA_H * self.In_H * SIZE_OF_WEIGHT
        self.addrT.new_layer_init(self.all_weights, w_par_size_c, out_size)
    
    def load_data(self, p_m_R: str):
        """从内存中载入数据
        :param p_m_R:   数据的寄存器名
        数据大小在一个tstep中记录
        """
        data_size = self.whole_tstep_bk
        return load_data_lib(p_m_R=p_m_R, size=data_size)
    
    def matrix_multiply(self, p_d_R: str, p_w_R: str, isNewVec: bool):
        """执行矩阵乘法
        每次计算数据量为数据矩阵的一列
        :param p_d_R:   数据的寄存器名
        """
        num_of_elements = self.In_H   # 执行计算的元素个数
        instr = (
            acc_lib(p_d_R, p_w_R, num_of_elements, isNewVec) +
            fire()
        )
        return instr
    
    def write_back(self, wb_addr_R: str):
        """将结果写回内存
        :param wb_addr_R:   写回地址的寄存器名
        """
        out_bk = self.valid_sa_h * SA_W / 8
        instr = store_output(wb_addr_R, size=out_bk)
        return instr
    
    def my_prim(self, i_weight_line):
        """ 一个任务的主要部分\n
            linear 的prim任务是： 载入数据，计算点乘 (sa_h, W) · (W, 1) = (sa_H, 1)\n
            其中sa_h <= SA_H； （权重已经又前面的core prim载入完毕）\n
            其中(W,1) 的值都是脉冲向量；有 N_T = T_LEN / SA_W   \n
            一次性输入 (W, SA_W)，实际上是同时和 SA_W个向量在计算 （作为一组TSTEP）\n
            在同一组TSTEP中的SA_W个比特信息都是一起处理的，所以这里看做1个，也就是(W,1)中的1

        :return:  指令list
        """
        # 每次load一个tstep的数据， 执行一次acc_0、acc(>=0次)、fire、store； 然后执行下一个tstep

        # 第一步： 创建迭代器（地址寄存器）
        # linear是 分割了权重行，重用了数据向量，所以数据都是从 偏移量0开始的
        data_offset = 0    
        mdata_begin_addr = self.addrT.m_logic_addr(data_offset, type_name='data_in')    # 数据的逻辑地址 in mem
        # data addr form memory; data to cache is CDataWrite_R
        data_from_m_it = MIterator('data_from_m', mdata_begin_addr, self.whole_tstep_bk)
        # acc_data = CIterator('data_to_c', CDataWrite_R_name, self.whole_tstep_bk)

        # 写回地址的迭代器:
        #   写回地址在out feature 的偏移量
        wb_addr_offset = i_weight_line * self.sa_line_byte   # 起点是偏移步长是SA_H行脉动阵列
        wb_addr_begin = self.addrT.m_logic_addr(wb_addr_offset, type_name='data_out')
        wb_it = MIterator('wb_addr', wb_addr_begin, self.out_whole_tstep_bk)    # prim内输出步长是整个tstep

        # 第二步： 编写指令
        # --第0tstep
        instrs = [
            # init iterators
            data_from_m_it.init_instr() +
            wb_it.init_instr() +
            # 第 0 tstep
            #   set_R_R: 在load之前保存CDataWrite的值, 即为acc数据的起始地址
            set_R_R('acc_data_R', CDataWrite_R_name) +
            self.load_data(p_m_R=data_from_m_it.name) +
            self.matrix_multiply(p_d_R='acc_data_R', p_w_R='x0', isNewVec=True) +  # acc_0
            self.write_back(wb_addr_R=wb_it.name) +
            # 更新迭代器的值
            data_from_m_it.iterate_1_instr() +
            wb_it.iterate_1_instr()
        ]
        # --后续的tstep用循环实现
        # ----创建1层循环： tstep循环
        tstep_cycle = Cycle('tstep', 1, N_T, 1)     # 第0tstep单独处理

        # ----循环内容: load, acc and fire, write back
        tstep_cycle.body_instrs.append(
            # 在load之前保存CDataWrite的值, 即为acc数据的起始地址
            set_R_R('acc_data_R', CDataWrite_R_name) +
            self.load_data(p_m_R=data_from_m_it.name) +
            self.matrix_multiply(p_d_R='acc_data_R', p_w_R='x0', isNewVec=False) +     # acc
            self.write_back(wb_addr_R=wb_it.name) +
            # 更新迭代器的值
            data_from_m_it.iterate_1_instr() +
            wb_it.iterate_1_instr()
        )
        # 第三步： 汇总指令
        instrs.extend(tstep_cycle.instrs())
        return instrs

    def load_weights(self, i_weight_line):
        """ 载入权重
        :param i_weight_line:    开始-卷积核编号
        :return:            指令序列
        """
        # 载入所有的卷积核(权重）
        data_offset = i_weight_line * self.weight_aline_bk      # offset =  前面已经载入过的权重的大小
        #  对应数据在mem中的地址
        p_m = self.addrT.m_logic_addr(data_offset, type_name='weight')       # 对应数据在mem中的地址
        w_size = self.valid_sa_h * self.weight_aline_bk

        instr = (
            # 设置计算部件SA的 有效脉冲阵列行数， 存在特定名字的寄存器中！
            set_R_val(VALID_SA_H_R_name, self.valid_sa_h) +
            set_R_val('p_weight_m_R', p_m) +
            load_weight_lib(p_m_R='p_weight_m_R', size=w_size)
        )
        return instr
    
    def main(self) -> ProcessPrim:
        """算子的主控函数"""
        # 计算每块权重所占空间，并为其开辟空间
        weight_par_size = SA_H * self.Weight_W * SIZE_OF_WEIGHT
        pro_prim = ProcessPrim(layer_init_lib(weight_par_size))
        
        i_weight_line = 0
        # 判断当前执行的权重行是否超出了
        while i_weight_line < self.Weight_H:
            # 每次取的权重行数为SA_H
            self.valid_sa_h = SA_H
            #当到最后几行时
            if i_weight_line + SA_H > self.Weight_H:
                self.valid_sa_h = self.Weight_H - i_weight_line
            
            core_prim = CorePrim(self.load_weights(i_weight_line))
            pro_prim.add_child(core_prim)
            
            cur_prim = Prim(self.my_prim(i_weight_line))
            core_prim.add_child(cur_prim)
            
            CorePrimEnd([], core_prim)
            i_weight_line += self.valid_sa_h
        
        ProcessPrimEnd([], pro_prim)
        self.addrT.layer_tail()
        return pro_prim