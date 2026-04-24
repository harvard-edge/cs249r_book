// Nabbed from lodash
export class SortedArray extends Array {
    #maxLength;
    #keyPath;
  
    constructor(maxLength = 0, keyPath) {
      super();
      this.#maxLength = maxLength;
      this.#keyPath = keyPath;
    }
  
    push() {
      throw new Error("Can't push on to a sorted array");
    }
  
    unshift() {
      throw new Error("Can't unshift on to a sorted array");
    }
  
    insert(value) {
      const array = this;
      const maxLength = this.#maxLength;
      let low = 0, high = array == null ? low : array.length;
  
      const accessor = typeof value == "object"
        ? (array, mid) => array[mid][this.#keyPath]
        : (array, mid) => array[mid];
      const resolvedValue = typeof value == "object" ? value[this.#keyPath] : value;
  
      while (low < high) {
        let mid = (low + high) >>> 1;
        let computed = accessor(array, mid);
  
        if ((computed !== null) & (computed >= resolvedValue)) {
          low = mid + 1;
        } else {
          high = mid;
        }
      }
  
      this.splice(high, 0, value);
  
      if (this.length > maxLength) {
        this.pop(); // Remove the last entry to make way for the new one
      }
    }
  }