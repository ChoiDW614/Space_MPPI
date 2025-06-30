from builtin_interfaces.msg import Time as MSG_Time

class Time():
    def __init__(self, sec: int = None, nanosec: int = None):
        if sec is None:
            sec = 0
        if nanosec is None:
            nanosec = 0

        self.__sec = sec
        self.__nanosec = nanosec
        self.__time = self.__sec + self.__nanosec * 1e-9

    @property
    def sec(self):
        return self.__sec
    
    @sec.setter
    def sec(self, sec):
        self.__sec = sec
        self._time()

    @property
    def nanosec(self):
        return self.__nanosec
    
    @sec.setter
    def nanosec(self, nanosec):
        self.__nanosec = nanosec
        self._time()

    @property
    def time(self):
        return self.__time

    @time.setter
    def time(self, args):
        if isinstance(args, MSG_Time):
            self.__sec = args.sec
            self.__nanosec = args.nanosec
            self._time()
        elif isinstance(args, float):
            self.__sec = int(args)
            self.__nanosec = int((args - int(args)) * 1e9)
            self._time()
        else:
            sec, nanosec = args
            self.__sec = sec
            self.__nanosec = nanosec
            self._time()


    def _time(self):
        self.__time = self.__sec + self.__nanosec * 1e-9

    def __repr__(self):
        return f"Time : {self.__time} sec"
    
    def __add__(self, ptime):
        return self.__time + ptime.__time
    
    def __sub__(self, ptime):
        return self.__time - ptime.__time
    