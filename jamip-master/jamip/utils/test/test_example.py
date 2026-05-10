# content of test_sample.py
def inc(x):
    return x + 1


def test_answer():
    assert inc(3) == 4


def _02(self,tmp_path_factory):
    d = tmp_path_factory.mktemp("demo")
    aa = d / '223.txt'
    aa.write_text('testfile')
    print(aa.as_posix())
    print('测试目录地址：%s'%d)

    txt = aa.read_text()
    print(txt)
    assert txt == 'testfile'
