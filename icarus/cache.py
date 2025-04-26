"""Cache implementation
"""
from collections import deque
class DataPacket(object):

    def __init__(self, content, lifetime, size):
        """
        Constructor
        """
        self.content = content 
        self.lifetime = lifetime
        self.size = size


class Cache(object):
    """
    Implementation of cache with LRU eviction policy
    """
    
    def __init__(self, max_size, slots_available):
        """
        Constructor
        """
        self.memory_space_left = max_size
        self.cache = deque(maxlen=int(slots_available))
    
    def store(self, dpacket):
        """
        Stores the content ID in the cache if not present yet. Otherwise, it
        pushes it on top of the cache 
        """
        if not self.has_content(dpacket) and self.memory_space_left > dpacket.size:

            self.memory_space_left -= dpacket.size
            self.cache.appendleft(dpacket)
            
    def has_content(self, content):
        """
        Return True if the cache contains the piece of content (and updates its
        internal status, e.g. push content on top of LRU stack) otherwise
        returns False
        """
        # search content over the list
        # if it has it push on top, otherwise return false
        try:
            self.cache.remove(content)
            self.cache.appendleft(content)
        except ValueError:
            return False
        return True

    def memeory_slots_left(self):
        """
        Return memory left in the cache node
        """

        try:
            return self.cache.maxlen - len(self.cache)
        except Exception as e:
            return False
        
    def remove_invalid_content(self, time):
        """
        Removes content whose life is over based on lifetime
        """

        try:
            for packet in self.cache:
                if time >= packet.lifetime: 
                    self.cache.remove(packet)
                    self.memory_space_left += packet.size
        except Exception as e:
            return e 
        

        