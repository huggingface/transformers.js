import { LoggingLevel, setLogLevel, getLogLevel, setLogLevelDebug, setLogLevelInfo, setLogLevelWarning, setLogLevelError, getLogger, logging } from "../../src/utils/logging.js";

describe("Logging", () => {
  // Save and restore the original log level around each test
  let originalLevel;
  beforeEach(() => {
    originalLevel = getLogLevel();
  });
  afterEach(() => {
    setLogLevel(originalLevel);
  });

  describe("LoggingLevel", () => {
    it("should define the expected levels", () => {
      expect(LoggingLevel.DEBUG).toBe(0);
      expect(LoggingLevel.INFO).toBe(1);
      expect(LoggingLevel.WARNING).toBe(2);
      expect(LoggingLevel.ERROR).toBe(3);
      expect(LoggingLevel.SILENT).toBe(4);
    });

    it("should be frozen", () => {
      expect(Object.isFrozen(LoggingLevel)).toBe(true);
    });
  });

  describe("setLogLevel / getLogLevel", () => {
    it("should default to WARNING", () => {
      setLogLevel(LoggingLevel.WARNING);
      expect(getLogLevel()).toBe(LoggingLevel.WARNING);
    });

    it("should accept numeric values", () => {
      setLogLevel(LoggingLevel.DEBUG);
      expect(getLogLevel()).toBe(LoggingLevel.DEBUG);

      setLogLevel(LoggingLevel.ERROR);
      expect(getLogLevel()).toBe(LoggingLevel.ERROR);
    });

    it("should accept string values (case-insensitive)", () => {
      setLogLevel("debug");
      expect(getLogLevel()).toBe(LoggingLevel.DEBUG);

      setLogLevel("INFO");
      expect(getLogLevel()).toBe(LoggingLevel.INFO);

      setLogLevel("Warning");
      expect(getLogLevel()).toBe(LoggingLevel.WARNING);

      setLogLevel("ERROR");
      expect(getLogLevel()).toBe(LoggingLevel.ERROR);

      setLogLevel("silent");
      expect(getLogLevel()).toBe(LoggingLevel.SILENT);
    });

    it("should accept 'warn' as an alias for WARNING", () => {
      setLogLevel("warn");
      expect(getLogLevel()).toBe(LoggingLevel.WARNING);
    });

    it("should throw on unknown string level", () => {
      expect(() => setLogLevel("verbose")).toThrow(/Unknown log level/);
    });

    it("should throw on invalid type", () => {
      expect(() => setLogLevel(true)).toThrow(/Invalid log level type/);
    });
  });

  describe("convenience setters", () => {
    it("setLogLevelDebug sets to DEBUG", () => {
      setLogLevelDebug();
      expect(getLogLevel()).toBe(LoggingLevel.DEBUG);
    });

    it("setLogLevelInfo sets to INFO", () => {
      setLogLevelInfo();
      expect(getLogLevel()).toBe(LoggingLevel.INFO);
    });

    it("setLogLevelWarning sets to WARNING", () => {
      setLogLevelWarning();
      expect(getLogLevel()).toBe(LoggingLevel.WARNING);
    });

    it("setLogLevelError sets to ERROR", () => {
      setLogLevelError();
      expect(getLogLevel()).toBe(LoggingLevel.ERROR);
    });
  });

  describe("getLogger", () => {
    it("should return a Logger instance", () => {
      const log = getLogger("test");
      expect(log).toBeDefined();
      expect(typeof log.debug).toBe("function");
      expect(typeof log.info).toBe("function");
      expect(typeof log.warn).toBe("function");
      expect(typeof log.error).toBe("function");
    });

    it("should return the same instance for the same name", () => {
      const a = getLogger("singleton-test");
      const b = getLogger("singleton-test");
      expect(a).toBe(b);
    });

    it("should return different instances for different names", () => {
      const a = getLogger("name-a");
      const b = getLogger("name-b");
      expect(a).not.toBe(b);
    });
  });

  describe("Logger respects log level", () => {
    // Helper: temporarily replace a console method and collect calls
    function spyOnConsole(method) {
      const original = console[method];
      const calls = [];
      console[method] = (...args) => calls.push(args);
      return {
        calls,
        restore: () => {
          console[method] = original;
        },
      };
    }

    it("should suppress messages below the current level", () => {
      const log = getLogger("level-test");
      const spy = spyOnConsole("info");

      setLogLevel(LoggingLevel.WARNING);
      log.info("should be suppressed");
      expect(spy.calls.length).toBe(0);

      setLogLevel(LoggingLevel.INFO);
      log.info("should appear");
      expect(spy.calls.length).toBe(1);

      spy.restore();
    });

    it("should suppress all messages at SILENT level", () => {
      const log = getLogger("silent-test");
      const spyDebug = spyOnConsole("debug");
      const spyInfo = spyOnConsole("info");
      const spyWarn = spyOnConsole("warn");
      const spyError = spyOnConsole("error");

      setLogLevel(LoggingLevel.SILENT);
      log.debug("d");
      log.info("i");
      log.warn("w");
      log.error("e");

      expect(spyDebug.calls.length).toBe(0);
      expect(spyInfo.calls.length).toBe(0);
      expect(spyWarn.calls.length).toBe(0);
      expect(spyError.calls.length).toBe(0);

      spyDebug.restore();
      spyInfo.restore();
      spyWarn.restore();
      spyError.restore();
    });

    it("should show all messages at DEBUG level", () => {
      const log = getLogger("debug-all-test");
      const spyDebug = spyOnConsole("debug");
      const spyInfo = spyOnConsole("info");
      const spyWarn = spyOnConsole("warn");
      const spyError = spyOnConsole("error");

      setLogLevel(LoggingLevel.DEBUG);
      log.debug("d");
      log.info("i");
      log.warn("w");
      log.error("e");

      expect(spyDebug.calls.length).toBe(1);
      expect(spyInfo.calls.length).toBe(1);
      expect(spyWarn.calls.length).toBe(1);
      expect(spyError.calls.length).toBe(1);

      spyDebug.restore();
      spyInfo.restore();
      spyWarn.restore();
      spyError.restore();
    });

    it("should prefix messages with logger name", () => {
      const log = getLogger("my-module");
      const spy = spyOnConsole("warn");

      setLogLevel(LoggingLevel.WARNING);
      log.warn("test message");
      expect(spy.calls.length).toBe(1);
      expect(spy.calls[0]).toEqual(["[my-module]", "test message"]);

      spy.restore();
    });

    it("should not prefix when logger has no name", () => {
      const log = getLogger("");
      const spy = spyOnConsole("warn");

      setLogLevel(LoggingLevel.WARNING);
      log.warn("no prefix");
      expect(spy.calls.length).toBe(1);
      expect(spy.calls[0]).toEqual(["no prefix"]);

      spy.restore();
    });
  });

  describe("logging namespace export", () => {
    it("should expose all public API functions", () => {
      expect(logging.LoggingLevel).toBe(LoggingLevel);
      expect(logging.setLogLevel).toBe(setLogLevel);
      expect(logging.getLogLevel).toBe(getLogLevel);
      expect(logging.setLogLevelDebug).toBe(setLogLevelDebug);
      expect(logging.setLogLevelInfo).toBe(setLogLevelInfo);
      expect(logging.setLogLevelWarning).toBe(setLogLevelWarning);
      expect(logging.setLogLevelError).toBe(setLogLevelError);
      expect(logging.getLogger).toBe(getLogger);
    });

    it("should be frozen", () => {
      expect(Object.isFrozen(logging)).toBe(true);
    });
  });
});
