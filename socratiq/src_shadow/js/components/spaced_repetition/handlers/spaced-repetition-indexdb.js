// src_shadow/js/libs/utils/spacedRepetitionIndexDBHandler.js

import { SpacedRepetitionStorageHandler } from "./storage-handler";
import { jsonrepair } from "jsonrepair";
// import { DuckAI } from "../../../../libs/agents/duck-ai.js";
import { DuckAI } from "../../../libs/agents/duck-ai-cloudflare.js";
import { getDBInstance } from "../../../libs/utils/indexDb.js";
// import { DB_CONFIGS } from "../../../configs/db_configs_one.js";

/*

  The class below implements every method of the original SQLite handler

  but using IndexedDB. All functionality is preserved. (Some methods such as

  exec() are now stubs because SQL statements do not apply.)

*/

export class SpacedRepetitionIndexDBHandler extends SpacedRepetitionStorageHandler {
  constructor(modal) {
    super(modal);
    this.modal = modal; // Ensure modal is set
    this.inMemoryData = {
      currentChapter: null,
      currentDeck: [],
      allTags: new Set(),
      reviewHistory: {},
    };

    this.saveQueue = [];

    this.processingQueue = false;

    this.currentSaveNotificationId = null;

    this.dbManager = null;

    this.isFirstLoad = true;

    this.useFallback = false;

    this.initPromise = null;

    this.initialDeck = false;
    // A minimal notification handler stub

    this.notificationHandler = {
      showLoadingState: (msg = "Loading...") => {
        console.log(msg);
        return Date.now();
      },

      updateNotification: (id, msg, type) => {
        console.log(`[${type}] ${msg}`);
      },

      removeLoadingState: (id) => {
        /* Remove notification */
      },
    };

    // Start initialization

    this.initPromise = this.initializeDatabase();
  }

  async initializeDatabase() {
    try {
      console.log("Initializing IndexedDB database...");

      // Try to get existing instance first
      this.dbManager = await getDBInstance();
      await this.verifyDatabasePersistence();
      return true;
    } catch (error) {
      console.error("Failed to initialize IndexedDB database:", error);
      this.useFallback = true;
      return false;
    }
  }

  // Add this helper method

  async ensureInitialized() {
    if (!this.dbManager?.db) {
      await this.initPromise;
    }
    return this.dbManager?.db != null;
  }

  // (No need to "create tables" separately in IndexedDB – they are defined in initDB.)

  async initializeTables() {
    return;
  }

  // Utility helper: perform a multi‑store transaction

  performTransaction(storeNames, mode, operation) {
    return new Promise((resolve, reject) => {
      if (!this.dbManager.db) {
        reject(new Error("Database not initialized"));

        return;
      }

      const transaction = this.dbManager.db.transaction(storeNames, mode);

      transaction.onerror = (event) => {
        console.error("Transaction error:", event.target.error);

        reject(event.target.error);
      };

      operation(transaction, resolve, reject);
    });
  }

  // ───────────── SAVE TO LOCAL STORAGE ─────────────

  async saveToLocalStorage(fullChapterSets = null) {
    try {
      console.log("📝 Starting save to IndexedDB...", {
        fullChapterSets: fullChapterSets
          ? Array.from(fullChapterSets.entries())
          : null,
        currentChapter: this.inMemoryData.currentChapter,
        currentDeck: this.inMemoryData.currentDeck,
        allTags: Array.from(this.inMemoryData.allTags),
        reviewHistory: this.inMemoryData.reviewHistory,
        flashcards: this.modal.flashcards,
      });

      if (fullChapterSets) {
        // Clear all stores first
        console.log("🗑️ Clearing stores...");
        await this.dbManager.performTransaction(
          "chapters",
          "readwrite",
          (store) => store.clear()
        );
        await this.dbManager.performTransaction("cards", "readwrite", (store) =>
          store.clear()
        );
        await this.dbManager.performTransaction(
          "card_tags",
          "readwrite",
          (store) => store.clear()
        );

        // Save each chapter and its cards
        for (const [chapter, cards] of fullChapterSets) {
          console.log(
            `📚 Processing chapter ${chapter} with ${cards.length} cards`
          );

          // Get existing chapter title from database, or use default
          const existingChapter = await this.dbManager.performTransaction(
            "chapters",
            "readonly",
            (store) => {
              return new Promise((resolve) => {
                const req = store.get(chapter);
                req.onsuccess = () => resolve(req.result);
                req.onerror = () => resolve(null);
              });
            }
          );
          
          // Use existing title if available, otherwise generate default
          const chapterTitle = existingChapter?.title || this.getChapterTitle(chapter);
          
          const chapterRecord = {
            chapter,
            title: chapterTitle,
            is_current:
              chapter === this.inMemoryData.currentChapter?.chapter ? 1 : 0,
          };
          console.log(`📖 Saving chapter record:`, chapterRecord);
          await this.dbManager.performTransaction(
            "chapters",
            "readwrite",
            (store) => store.put(chapterRecord) // Use put() instead of add() to allow updates
          );

          // Update current_chapter store if this is the current chapter
          if (chapterRecord.is_current === 1) {
             await this.dbManager.performTransaction(
                "current_chapter",
                "readwrite",
                (store) => {
                    store.clear(); // Clear old current chapter
                    return store.add({
                        chapter: chapter,
                        title: chapterTitle
                    });
                }
             );
          }

          // Save cards
          for (const card of cards) {
            const processedCard = {
              ...card,
              chapter: chapter,
              created: card.created || new Date().toISOString(),
            };
            console.log(`💳 Adding card:`, {
              id: processedCard.id,
              chapter: processedCard.chapter,
              created: processedCard.created,
            });
            await this.dbManager.performTransaction(
              "cards",
              "readwrite",
              (store) => store.add(processedCard)
            );

            // Save tags
            if (card.tags && Array.isArray(card.tags)) {
              for (const tag of card.tags) {
                const tagRecord = {
                  id: `${card.id}_${tag}`,
                  card_id: card.id,
                  tag,
                };
                console.log(`🏷️ Adding tag:`, tagRecord);
                await this.dbManager.performTransaction(
                  "card_tags",
                  "readwrite",
                  (store) => store.add(tagRecord)
                );
              }
            }
          }

          // Update modal's chapter sets
          if (this.modal?.chapterSets) {
            console.log(
              `🔄 Updating modal chapter sets for chapter ${chapter}`
            );
            this.modal.chapterSets.set(chapter, cards);
          }
        }
        // } else if (this.inMemoryData.currentChapter) {
        //     const chapter = this.inMemoryData.currentChapter.chapter;
        //     console.log(`📝 Saving current chapter ${chapter}`);

        //     // Delete existing cards for this chapter
        //     await this.dbManager.performTransaction("cards", "readwrite", store => {
        //         const index = store.index("chapter");
        //         const range = IDBKeyRange.only(chapter);
        //         return index.openCursor(range);
        //     });
      } else if (this.inMemoryData.currentChapter) {
        const chapter = this.inMemoryData.currentChapter.chapter;
        
        // Ensure chapter exists in chapters table before saving cards
        // FIXED: performTransaction expects a request, not a Promise
        const chapterResult = await this.dbManager.performTransaction(
          "chapters",
          "readonly",
          (store) => {
            const req = store.get(chapter);
            // Wrap onsuccess to add logging, but let performTransaction handle resolution
            const originalOnSuccess = req.onsuccess;
            req.onsuccess = function() {
              if (originalOnSuccess) originalOnSuccess.call(this);
            };
            req.onerror = function() {
              console.error(`Chapter exists check error`);
            };
            return req;
          }
        );
        const chapterExists = !!chapterResult;

        if (!chapterExists) {
          // Create chapter record if it doesn't exist
          const chapterRecord = {
            chapter: chapter,
            title: this.inMemoryData.currentChapter.title || this.getChapterTitle(chapter),
            is_current: 1,
          };
          await this.dbManager.performTransaction(
            "chapters",
            "readwrite",
            (store) => store.put(chapterRecord)
          );
          
          // Also update current_chapter store
          await this.dbManager.performTransaction(
            "current_chapter",
            "readwrite",
            (store) => {
                store.clear();
                return store.add({
                    chapter: chapter,
                    title: chapterRecord.title
                });
            }
          );
          
          // Update other chapters to not current
          // FIXED: Return request directly, not a Promise
          const allChapters = await this.dbManager.performTransaction(
            "chapters",
            "readonly",
            (store) => {
              const req = store.getAll();
              const originalOnSuccess = req.onsuccess;
              req.onsuccess = function() {
                if (originalOnSuccess) originalOnSuccess.call(this);
              };
              req.onerror = function() {
                console.error(`Error getting chapters`);
              };
              return req;
            }
          ) || [];
          
          for (const ch of allChapters) {
            if (ch.chapter !== chapter) {
              ch.is_current = 0;
              await this.dbManager.performTransaction(
                "chapters",
                "readwrite",
                (store) => store.put(ch)
              );
            }
          }
        }

        // First, get all existing cards for this chapter
        const existingCards = await this.dbManager.performTransaction(
          "cards",
          "readonly",
          (store) => {
            const index = store.index("chapter");
            const req = index.getAll(chapter);
            req.onerror = () => console.error(`Error getting existing cards:`, req.error);
            return req;
          }
        );

        // Delete cards that are no longer in currentDeck
        const currentCardIds = new Set(
          this.inMemoryData.currentDeck.map((card) => card.id)
        );
        
        let deletedCount = 0;
        for (const existingCard of existingCards) {
          if (!currentCardIds.has(existingCard.id)) {
            await this.dbManager.performTransaction(
              "cards",
              "readwrite",
              (store) => {
                const req = store.delete(existingCard.id);
                req.onerror = () => console.error(`Error deleting card ${existingCard.id}:`, req.error);
                return req;
              }
            );
            deletedCount++;
            // TEMPORARILY DISABLED: Tag deletion is causing hangs
            // TODO: Fix tag deletion properly - tags will be orphaned but won't cause issues
          }
        }

        // Save current deck
        let savedCardCount = 0;
        for (const card of this.inMemoryData.currentDeck) {
          const processedCard = {
            ...card,
            chapter: chapter,
            created: card.created || new Date().toISOString(),
          };
          
          await this.dbManager.performTransaction(
            "cards",
            "readwrite",
            (store) => {
              const req = store.put(processedCard);
              req.onerror = () => console.error(`Error saving card ${card.id}:`, req.error);
              return req;
            }
          );

          // Save tags
          if (card.tags && Array.isArray(card.tags) && card.tags.length > 0) {
            for (let tagIndex = 0; tagIndex < card.tags.length; tagIndex++) {
              const tag = card.tags[tagIndex];
              await this.dbManager.performTransaction(
                "card_tags",
                "readwrite",
                (store) => {
                  const req = store.put({
                    id: `${card.id}_${tag}`,
                    card_id: card.id,
                    tag,
                  });
                  req.onerror = () => console.error(`Error saving tag "${tag}":`, req.error);
                  return req;
                }
              );
            }
          }
          savedCardCount++;
        }
      }

      console.log(`📝 Save to IndexedDB completed successfully!`);
      window.dispatchEvent(
        new CustomEvent("sr-save-card-completed", {
          detail: { success: true },
        })
      );
      return true;
    } catch (error) {
      console.error(`📝 [SAVE STEP ERROR] Failed to save to IndexedDB:`, error);
      console.error(`📝 [SAVE STEP ERROR] Error stack:`, error.stack);
      console.error(`📝 [SAVE STEP ERROR] Error name:`, error.name);
      console.error(`📝 [SAVE STEP ERROR] Error message:`, error.message);
      // Only fallback to localStorage if IndexedDB is completely unavailable
      // Don't fallback for data errors - throw them instead
      if (this.useFallback && !this.dbManager?.db) {
        console.warn("⚠️ IndexedDB unavailable, falling back to localStorage");
        return super.saveToLocalStorage(fullChapterSets);
      }
      throw error;
    }
  }

  purgeCardFromMemory(cardId) {
    console.log("🧹 Cleaning up card from memory:", cardId);

    console.log("i am current deck inMemoryData MODAL", this.inMemoryData);

    // Remove from current deck
    if (this.inMemoryData.currentDeck) {
      this.inMemoryData.currentDeck = this.inMemoryData.currentDeck.filter(
        (card) => card.id !== cardId
      );
    }

    // Remove from chapter sets if they exist
    if (this.modal?.chapterSets) {
      for (const [chapter, cards] of this.modal.chapterSets) {
        this.modal.chapterSets.set(
          chapter,
          cards.filter((card) => card.id !== cardId)
        );
      }
    }

    // Recalculate all tags
    const allTags = new Set();
    this.inMemoryData.currentDeck.forEach((card) => {
      if (card.tags && Array.isArray(card.tags)) {
        card.tags.forEach((tag) => allTags.add(tag));
      }
    });
    this.inMemoryData.allTags = allTags;

    console.log("✨ Memory cleanup completed");
  }

  // ───────────── INSERT CARD ─────────────

  async insertCard(card, chapter) {
    // if (this.useFallback) {
    //   return super.insertCard(card, chapter);
    // }

    try {
      await this.performTransaction(
        ["cards", "card_tags"],
        "readwrite",
        (tx, resolve, reject) => {
          const cardStore = tx.objectStore("cards");

          const tagStore = tx.objectStore("card_tags");

          const req = cardStore.add(card);

          req.onsuccess = () => {
            if (card.tags && Array.isArray(card.tags)) {
              card.tags.forEach((tag) => {
                tagStore.add({
                  id: `${card.id}_${tag}`,
                  card_id: card.id,
                  tag,
                });
              });
            }

            resolve(true);
          };

          req.onerror = () => reject(req.error);
        }
      );
    } catch (error) {
      console.error("Error inserting card:", error);

      throw error;
    }
  }

  // ───────────── GET ALL CHAPTERS ─────────────

  async getAllChapters() {
    // if (this.useFallback) {
    //   return super.getAllChapters();
    // }

    try {
      await this.ensureInitialized();
      const chapters = await new Promise((resolve, reject) => {
        const tx = this.dbManager.db.transaction("chapters", "readonly");

        const req = tx.objectStore("chapters").getAll();

        req.onsuccess = (event) => resolve(event.target.result);

        req.onerror = (event) => reject(event.target.error);
      });

      // For each chapter, count its cards.

      const results = await Promise.all(
        chapters.map(async (chapter) => {
          const count = await new Promise((resolve, reject) => {
            const tx = this.dbManager.db.transaction("cards", "readonly");

            const index = tx.objectStore("cards").index("chapter");

            const req = index.count(IDBKeyRange.only(chapter.chapter));

            req.onsuccess = (e) => resolve(e.target.result);

            req.onerror = (e) => reject(e.target.error);
          });

          return {
            chapter: chapter.chapter,

            title: chapter.title || `Chapter ${chapter.chapter}`,

            cardCount: count,
          };
        })
      );

      return results;
    } catch (error) {
      console.error("Failed to get all chapters:", error);

      if (this.useFallback) {
        return super.getAllChapters();
      }

      throw error;
    }
  }

  // ───────────── SWITCH ACTIVE CHAPTER ─────────────

  async switchActiveChapter(chapterNum) {
    try {
      await this.ensureInitialized();

      await this.performTransaction(
        ["chapters", "cards"],
        "readwrite",
        async (tx, resolve, reject) => {
          try {
            // Update chapters
            const chapterStore = tx.objectStore("chapters");

            // Set all chapters to not current
            const chaptersRequest = chapterStore.getAll();
            chaptersRequest.onsuccess = () => {
              const chapters = chaptersRequest.result;
              chapters.forEach((chapter) => {
                chapter.is_current = chapter.chapter === chapterNum ? 1 : 0;
                chapterStore.put(chapter);
              });
            };

            // Get cards for the chapter
            const cardStore = tx.objectStore("cards");
            const chapterIndex = cardStore.index("chapter");
            const cardsRequest = chapterIndex.getAll(chapterNum);

            cardsRequest.onsuccess = () => {
              const cards = cardsRequest.result;

              // Update in-memory data
              this.inMemoryData.currentChapter = {
                chapter: chapterNum,
                title: this.getChapterTitle(chapterNum),
              };
              this.inMemoryData.currentDeck = cards;

              // Update modal's chapter sets if they exist
              if (this.modal?.chapterSets) {
                this.modal.chapterSets.set(chapterNum, cards);
              }

              resolve({
                currentChapter: this.inMemoryData.currentChapter,
                currentDeck: this.inMemoryData.currentDeck,
              });
            };

            cardsRequest.onerror = (error) => {
              console.error("Failed to get cards for chapter:", error);
              reject(error);
            };
          } catch (error) {
            console.error("Error in switch chapter transaction:", error);
            reject(error);
          }
        }
      );
    } catch (error) {
      console.error("Failed to switch active chapter:", error);
      if (this.useFallback) {
        return super.switchActiveChapter(chapterNum);
      }
      throw error;
    }
  }

  // ───────────── GET CURRENT CHAPTER DATA ─────────────

  getCurrentChapterData() {
    return {
      chapter: this.inMemoryData.currentChapter,

      deck: this.inMemoryData.currentDeck,

      stats: this.getCurrentChapterStats(),

      tags: Array.from(this.inMemoryData.allTags),
    };
  }

  // ───────────── GET ALL TAGS ─────────────

  async getAllTags() {
    // if (this.useFallback) {
    //   return super.getAllTags();
    // }

    try {
      const tags = await new Promise((resolve, reject) => {
        const tx = this.dbManager.db.transaction("card_tags", "readonly");

        const req = tx.objectStore("card_tags").getAll();

        req.onsuccess = (e) => {
          const allTags = e.target.result.map((r) => r.tag);

          resolve([...new Set(allTags)]);
        };

        req.onerror = (e) => reject(e.target.error);
      });

      return tags;
    } catch (error) {
      console.error("Failed to get all tags:", error);

      if (this.useFallback) {
        return super.getAllTags();
      }

      return [];
    }
  }

  // ───────────── GET CARDS BY TAG ─────────────

  async getCardsByTag(tag) {
    // if (this.useFallback) {
    //   return super.getCardsByTag(tag);
    // }

    try {
      const cardIds = await new Promise((resolve, reject) => {
        const tx = this.dbManager.db.transaction("card_tags", "readonly");

        const index = tx.objectStore("card_tags").index("tag");

        const req = index.getAll(IDBKeyRange.only(tag));

        req.onsuccess = (e) => {
          const ids = e.target.result.map((r) => r.card_id);

          resolve(ids);
        };

        req.onerror = (e) => reject(e.target.error);
      });

      const cards = await Promise.all(
        cardIds.map((cardId) => {
          return new Promise((resolve, reject) => {
            const tx = this.dbManager.db.transaction("cards", "readonly");

            const req = tx.objectStore("cards").get(cardId);

            req.onsuccess = (e) => resolve(e.target.result);

            req.onerror = (e) => reject(e.target.error);
          });
        })
      );

      // For each card, attach its tags.

      for (let card of cards) {
        card.tags = await new Promise((resolve, reject) => {
          const tx = this.dbManager.db.transaction("card_tags", "readonly");

          const index = tx.objectStore("card_tags").index("card_id");

          const req = index.getAll(IDBKeyRange.only(card.id));

          req.onsuccess = (e) => resolve(e.target.result.map((r) => r.tag));

          req.onerror = (e) => reject(e.target.error);
        });
      }

      return cards;
    } catch (error) {
      console.error("Failed to get cards by tag:", error);

      if (this.useFallback) {
        return super.getCardsByTag(tag);
      }

      return [];
    }
  }

  // ───────────── QUEUE SAVE & PROCESS QUEUE ─────────────

  queueSave() {
    if (!this.currentSaveNotificationId) {
      this.currentSaveNotificationId =
        this.notificationHandler.showLoadingState("Saving...");
    }

    this.saveQueue.push({
      timestamp: Date.now(),
      notificationId: this.currentSaveNotificationId,
    });

    this.debouncedProcessQueue();
  }

  debouncedProcessQueue = this.debounce(() => {
    this.processQueue();
  }, 500);

  async processQueue() {
    if (this.processingQueue || this.saveQueue.length === 0) return;

    this.processingQueue = true;

    this.saveQueue = [];

    try {
      if (!this.currentSaveNotificationId) {
        this.currentSaveNotificationId =
          this.notificationHandler.showLoadingState("Saving...");
      }

      await this.saveToLocalStorage();

      this.notificationHandler.updateNotification(
        this.currentSaveNotificationId,
        "Saved successfully!",
        "success"
      );
    } catch (error) {
      console.error("Failed to process save queue:", error);

      this.notificationHandler.updateNotification(
        this.currentSaveNotificationId,
        "Failed to save",
        "error"
      );
    } finally {
      this.currentSaveNotificationId = null;

      this.processingQueue = false;

      if (this.saveQueue.length > 0) {
        this.debouncedProcessQueue();
      }
    }
  }

  // ───────────── INITIALIZE FROM STORAGE ─────────────

  async initializeFromStorage() {
    // if (this.useFallback) {
    //   return super.initializeFromStorage();
    // }

    try {
      // Get current chapter from store.

      let currentChapter = await new Promise((resolve, reject) => {
        const tx = this.dbManager.db.transaction("current_chapter", "readonly");

        const req = tx.objectStore("current_chapter").getAll();

        req.onsuccess = (e) => resolve(e.target.result[0]);

        req.onerror = (e) => reject(e.target.error);
      });

      // Fallback: If no current chapter in separate store, check chapters table for is_current flag
      if (!currentChapter) {
        currentChapter = await new Promise((resolve, reject) => {
            const tx = this.dbManager.db.transaction("chapters", "readonly");
            const req = tx.objectStore("chapters").getAll();
            
            req.onsuccess = (e) => {
                const chapters = e.target.result;
                const active = chapters.find(c => c.is_current === 1);
                resolve(active);
            };
            
            req.onerror = (e) => reject(e.target.error);
        });
      }

      if (currentChapter) {
        this.inMemoryData.currentChapter = currentChapter;

        const cards = await new Promise((resolve, reject) => {
          const tx = this.dbManager.db.transaction("cards", "readonly");

          const index = tx.objectStore("cards").index("chapter");

          const req = index.getAll(IDBKeyRange.only(currentChapter.chapter));

          req.onsuccess = (e) => resolve(e.target.result);

          req.onerror = (e) => reject(e.target.error);
        });

        for (let card of cards) {
          card.tags = await new Promise((resolve, reject) => {
            const tx = this.dbManager.db.transaction("card_tags", "readonly");

            const index = tx.objectStore("card_tags").index("card_id");

            const req = index.getAll(IDBKeyRange.only(card.id));

            req.onsuccess = (e) => resolve(e.target.result.map((r) => r.tag));

            req.onerror = (e) => reject(e.target.error);
          });
        }

        this.inMemoryData.currentDeck = cards;
      }

      const tags = await this.getAllTags();

      this.inMemoryData.allTags = new Set(tags);

      const reviewHistory = await new Promise((resolve, reject) => {
        const tx = this.dbManager.db.transaction("review_history", "readonly");

        const req = tx.objectStore("review_history").getAll();

        req.onsuccess = (e) => {
          const history = e.target.result.reduce((acc, rec) => {
            acc[rec.date] = rec.count;
            return acc;
          }, {});

          resolve(history);
        };

        req.onerror = (e) => reject(e.target.error);
      });

      this.inMemoryData.reviewHistory = reviewHistory;

      if (!this.inMemoryData.currentChapter) {
        await this.initializeFromStorage();
      }

      return {
        currentChapter: this.inMemoryData.currentChapter,

        chapterSets: await this.getAllChapters(),

        flashcards: this.inMemoryData.currentDeck,

        allTags: Array.from(this.inMemoryData.allTags),

        reviewHistory: this.inMemoryData.reviewHistory,

        stats: this.getCurrentChapterStats(),
      };
    } catch (error) {
      console.error("Failed to load from IndexedDB:", error);

      if (this.useFallback) {
        return super.initializeFromStorage();
      }

      this.inMemoryData = {
        currentChapter: null,
        currentDeck: [],
        allTags: new Set(),
        reviewHistory: {},
      };

      return {
        currentChapter: null,
        chapterSets: [],
        flashcards: [],
        allTags: [],
        reviewHistory: {},
      };
    }
  }

  // ───────────── MERGE IMPORTED DATA ─────────────

  async mergeImportedData(importedData) {
    // if (this.useFallback) {
    //   return super.mergeImportedData(importedData);
    // }

    try {
      if (!importedData.chapter_card_sets) {
        throw new Error("Invalid import data structure");
      }

      await this.performTransaction(
        ["chapters", "cards", "card_tags", "current_chapter"],
        "readwrite",
        (tx, resolve, reject) => {
          try {
            importedData.chapter_card_sets.forEach((chapterSet) => {
              const chapterStore = tx.objectStore("chapters");

              const chapterRecord = {
                chapter: chapterSet.chapter,
                title: this.getChapterTitle(chapterSet.chapter),
              };

              chapterStore.put(chapterRecord);

              const cardStore = tx.objectStore("cards");

              const tagStore = tx.objectStore("card_tags");

              chapterSet.cards.forEach((card) => {
                cardStore.put(card);

                if (card.tags && Array.isArray(card.tags)) {
                  card.tags.forEach((tag) => {
                    tagStore.put({
                      id: `${card.id}_${tag}`,
                      card_id: card.id,
                      tag,
                    });
                  });
                }
              });
            });

            if (importedData.current_chapter) {
              const currentStore = tx.objectStore("current_chapter");

              currentStore.clear();

              currentStore.add(importedData.current_chapter);
            }

            tx.oncomplete = () => resolve();
          } catch (err) {
            reject(err);
          }
        }
      );

      if (this.inMemoryData.currentChapter) {
        await this.switchActiveChapter(
          this.inMemoryData.currentChapter.chapter
        );
      }

      return true;
    } catch (error) {
      console.error("Error merging imported data:", error);

      throw error;
    }
  }

  // ───────────── UPDATE CARD ─────────────

  async updateCard(newCard) {
    // if (this.useFallback) {
    //   return super.updateCard(newCard);
    // }

    if (!this.inMemoryData.currentChapter) {
      console.error("No current chapter selected");

      throw new Error("No current chapter selected");
    }

    try {
      await this.performTransaction(
        ["cards", "card_tags"],
        "readwrite",
        (tx, resolve, reject) => {
          const cardStore = tx.objectStore("cards");

          cardStore.put(newCard);

          const tagStore = tx.objectStore("card_tags");

          // Delete existing tags for this card.

          const index = tagStore.index("card_id");

          const req = index.openCursor(IDBKeyRange.only(newCard.id));

          req.onsuccess = (event) => {
            const cursor = event.target.result;

            if (cursor) {
              tagStore.delete(cursor.primaryKey);

              cursor.continue();
            }
          };

          if (newCard.tags && Array.isArray(newCard.tags)) {
            newCard.tags.forEach((tag) => {
              tagStore.add({
                id: `${newCard.id}_${tag}`,
                card_id: newCard.id,
                tag,
              });

              this.inMemoryData.allTags.add(tag);
            });
          }

          tx.oncomplete = () => resolve();
        }
      );

      const cardIndex = this.inMemoryData.currentDeck.findIndex(
        (c) => c.id === newCard.id
      );

      if (cardIndex >= 0) {
        this.inMemoryData.currentDeck[cardIndex] = newCard;
      } else {
        this.inMemoryData.currentDeck.push(newCard);
      }

      return {
        currentDeck: this.inMemoryData.currentDeck,

        allTags: Array.from(this.inMemoryData.allTags),

        stats: this.getCurrentChapterStats(),

        currentChapter: this.inMemoryData.currentChapter,
      };
    } catch (error) {
      console.error("Failed to update card:", error);

      throw error;
    }
  }

  // ───────────── CREATE NEW CARD ─────────────

  async createNewCard(question, answer, tags = []) {
    // if (this.useFallback) {
    //   return super.createNewCard(question, answer, tags);
    // }

    if (!this.inMemoryData.currentChapter) {
      console.error("No current chapter selected");

      throw new Error("No current chapter selected");
    }

    const newCard = {
      id: crypto.randomUUID(),

      chapter: this.inMemoryData.currentChapter.chapter,

      question,

      answer,

      created: new Date().toISOString(),

      repetitions: 0,

      easeFactor: 2.5,

      interval: 0,

      nextReviewDate: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(),

      lastReviewQuality: 0,

      tags,
    };

    try {
      await this.performTransaction(
        ["cards", "card_tags"],
        "readwrite",
        (tx, resolve, reject) => {
          const cardStore = tx.objectStore("cards");

          cardStore.add(newCard);

          if (tags && tags.length > 0) {
            const tagStore = tx.objectStore("card_tags");

            tags.forEach((tag) => {
              tagStore.add({
                id: `${newCard.id}_${tag}`,
                card_id: newCard.id,
                tag,
              });
            });
          }

          tx.oncomplete = () => resolve();
        }
      );

      this.inMemoryData.currentDeck.push(newCard);

      this.emitDataUpdated && this.emitDataUpdated();

      return newCard;
    } catch (error) {
      console.error("Failed to create new card:", error);

      throw error;
    }
  }

  // ───────────── GET CURRENT CHAPTER STATS ─────────────

  getCurrentChapterStats() {
    if (!this.inMemoryData.currentDeck) {
      return { total: 0, learned: 0, percentage: 0 };
    }

    const total = this.inMemoryData.currentDeck.length;

    const learned = this.inMemoryData.currentDeck.filter(
      (card) =>
        card.repetitions > 0 &&
        card.nextReviewDate &&
        new Date(card.nextReviewDate) > new Date()
    ).length;

    return {
      total,
      learned,
      percentage: total > 0 ? Math.round((learned / total) * 100) : 0,
    };
  }

  // ───────────── CHECK & PRUNE DUPLICATES ─────────────

  async checkAndPruneDuplicates() {
    // if (this.useFallback) {
    //   return super.checkAndPruneDuplicates();
    // }

    const duplicates = { byId: [], byContent: [] };

    const seenIds = new Set();

    const seenContent = new Set();

    const uniqueCards = [];

    this.inMemoryData.currentDeck.forEach((card) => {
      const contentSignature = `${card.question}:${card.answer}`;

      if (seenIds.has(card.id)) {
        duplicates.byId.push(card);
      } else if (seenContent.has(contentSignature)) {
        duplicates.byContent.push(card);
      } else {
        seenIds.add(card.id);

        seenContent.add(contentSignature);

        uniqueCards.push(card);
      }
    });

    if (duplicates.byId.length > 0 || duplicates.byContent.length > 0) {
      await this.performTransaction(
        ["cards", "card_tags"],
        "readwrite",
        (tx, resolve, reject) => {
          const cardStore = tx.objectStore("cards");

          const tagStore = tx.objectStore("card_tags");

          [...duplicates.byId, ...duplicates.byContent].forEach((card) => {
            cardStore.delete(card.id);

            const index = tagStore.index("card_id");

            const req = index.openCursor(IDBKeyRange.only(card.id));

            req.onsuccess = (event) => {
              const cursor = event.target.result;

              if (cursor) {
                tagStore.delete(cursor.primaryKey);

                cursor.continue();
              }
            };
          });

          tx.oncomplete = () => resolve();
        }
      );

      this.inMemoryData.currentDeck = uniqueCards;

      this.dispatchEvent &&
        this.dispatchEvent(
          new CustomEvent("data-updated", {
            detail: {
              type: "duplicates-removed",
              removedCount: {
                byId: duplicates.byId.length,
                byContent: duplicates.byContent.length,
              },
            },
          })
        );
    }

    return {
      uniqueCards,
      duplicates,
      totalRemoved: duplicates.byId.length + duplicates.byContent.length,
    };
  }

  // ───────────── IMPORT FROM JSON ─────────────

  async importFromJSON(jsonData) {
    // if (this.useFallback) {
    //   return super.importFromJSON(jsonData);
    // }

    try {
      await this.performTransaction(
        ["cards", "card_tags", "chapters", "current_chapter", "review_history"],
        "readwrite",
        (tx, resolve, reject) => {
          [
            "cards",
            "card_tags",
            "chapters",
            "current_chapter",
            "review_history",
          ].forEach((storeName) => {
            tx.objectStore(storeName).clear();
          });

          if (jsonData.chapter_progress_data) {
            jsonData.chapter_progress_data.forEach((chapter) => {
              tx.objectStore("chapters").add({
                chapter: chapter.chapter,

                title: chapter.title,

                quizzesTaken: chapter.quizzesTaken || 0,

                totalQuizzes: chapter.totalQuizzes || 0,

                passedQuizzes: chapter.passedQuizzes || 0,
              });
            });
          }

          if (jsonData.chapter_card_sets) {
            jsonData.chapter_card_sets.forEach((chapterSet) => {
              chapterSet.cards.forEach((card) => {
                tx.objectStore("cards").add(card);

                if (card.tags && card.tags.length > 0) {
                  card.tags.forEach((tag) => {
                    tx.objectStore("card_tags").add({
                      id: `${card.id}_${tag}`,
                      card_id: card.id,
                      tag,
                    });
                  });
                }
              });
            });
          }

          if (jsonData.current_chapter) {
            tx.objectStore("current_chapter").add(jsonData.current_chapter);
          }

          if (jsonData.review_activity) {
            Object.entries(jsonData.review_activity).forEach(
              ([date, count]) => {
                tx.objectStore("review_history").add({ date, count });
              }
            );
          }

          tx.oncomplete = () => resolve();
        }
      );

      await this.initializeFromStorage();

      return true;
    } catch (error) {
      console.error("Failed to import JSON data:", error);

      throw error;
    }
  }

  // ───────────── EXPORT TO JSON ─────────────

  async exportToJSON() {
    // if (this.useFallback) {
    //   return super.exportToJSON();
    // }

    try {
      const cards = await new Promise((resolve, reject) => {
        const tx = this.dbManager.db.transaction("cards", "readonly");

        const req = tx.objectStore("cards").getAll();

        req.onsuccess = (e) => resolve(e.target.result);

        req.onerror = (e) => reject(e.target.error);
      });

      for (let card of cards) {
        card.tags = await new Promise((resolve, reject) => {
          const tx = this.dbManager.db.transaction("card_tags", "readonly");

          const index = tx.objectStore("card_tags").index("card_id");

          const req = index.getAll(IDBKeyRange.only(card.id));

          req.onsuccess = (e) => resolve(e.target.result.map((r) => r.tag));

          req.onerror = (e) => reject(e.target.error);
        });
      }

      const currentChapter = await new Promise((resolve, reject) => {
        const tx = this.dbManager.db.transaction("current_chapter", "readonly");

        const req = tx.objectStore("current_chapter").getAll();

        req.onsuccess = (e) => resolve(e.target.result[0]);

        req.onerror = (e) => reject(e.target.error);
      });

      // Organize cards by chapter.

      const chapterCardSets = [];

      const cardsByChapter = new Map();

      cards.forEach((card) => {
        if (!cardsByChapter.has(card.chapter)) {
          cardsByChapter.set(card.chapter, []);
        }

        cardsByChapter.get(card.chapter).push(card);
      });

      for (const [chapter, cards] of cardsByChapter) {
        chapterCardSets.push({ chapter: parseInt(chapter), cards });
      }

      const exportData = {
        chapter_card_sets: chapterCardSets,

        current_chapter: currentChapter || null,

        exported_at: new Date().toISOString(),
      };

      return exportData;
    } catch (error) {
      console.error("Failed to export data to JSON:", error);

      throw error;
    }
  }

  // ───────────── EXEC (Stub – not applicable in IndexedDB) ─────────────

  async exec(query, params = []) {
    console.warn("exec method is not applicable in IndexedDB implementation");

    return;
  }

  // ───────────── DEBUG DATABASE ─────────────

  async debugDatabase() {
    try {
      console.log("=== DATABASE DEBUG INFO ===");

      for (const storeName of [
        "review_history",
        "chapters",
        "cards",
        "card_tags",
        "current_chapter",
      ]) {
        const data = await new Promise((resolve, reject) => {
          const tx = this.dbManager.db.transaction(storeName, "readonly");

          const req = tx.objectStore(storeName).getAll();

          req.onsuccess = (e) => resolve(e.target.result);

          req.onerror = (e) => reject(e.target.error);
        });

        console.log(`Store: ${storeName}`, data);
      }

      console.log("=== END DEBUG INFO ===");
    } catch (error) {
      console.error("Debug error:", error);
    }
  }

  // ───────────── MIGRATE CURRENT CHAPTER ─────────────

  async migrateCurrentChapter() {
    try {
      const currentChapterData = localStorage.getItem("current_chapter");

      if (!currentChapterData) {
        console.log("No current chapter to migrate");

        return null;
      }

      const currentChapter = JSON.parse(currentChapterData);

      const existing = await new Promise((resolve, reject) => {
        const tx = this.dbManager.db.transaction("current_chapter", "readonly");

        const req = tx
          .objectStore("current_chapter")
          .get(currentChapter.chapter);

        req.onsuccess = (e) => resolve(e.target.result);

        req.onerror = (e) => reject(e.target.error);
      });

      if (existing) {
        this.inMemoryData.currentChapter = existing;

        return existing;
      }

      await this.performTransaction(
        ["current_chapter"],
        "readwrite",
        (tx, resolve, reject) => {
          const store = tx.objectStore("current_chapter");

          store.clear();

          store.add(currentChapter);

          tx.oncomplete = () => resolve();
        }
      );

      this.inMemoryData.currentChapter = currentChapter;

      return currentChapter;
    } catch (error) {
      console.error("Failed to migrate current chapter:", error);

      return null;
    }
  }

  // ───────────── GET REVIEW COUNT ─────────────

  async getReviewCount(date = null) {
    try {
      const targetDate = date || new Date().toISOString().split("T")[0];

      const count = await new Promise((resolve, reject) => {
        const tx = this.dbManager.db.transaction("review_history", "readonly");

        const req = tx.objectStore("review_history").get(targetDate);

        req.onsuccess = (e) => {
          const record = e.target.result;

          resolve(record ? record.count : 0);
        };

        req.onerror = (e) => reject(e.target.error);
      });

      return count;
    } catch (error) {
      console.error("Error getting review count:", error);

      return 0;
    }
  }

  // ───────────── DEBUG REVIEW HISTORY ─────────────

  async debugReviewHistory() {
    try {
      const history = await new Promise((resolve, reject) => {
        const tx = this.dbManager.db.transaction("review_history", "readonly");

        const req = tx.objectStore("review_history").getAll();

        req.onsuccess = (e) => resolve(e.target.result);

        req.onerror = (e) => reject(e.target.error);
      });

      console.table(history);

      return history;
    } catch (error) {
      console.error("Error debugging review history:", error);

      return [];
    }
  }

  // ───────────── VERIFY DATABASE PERSISTENCE ─────────────

  async verifyDatabasePersistence() {
    try {
      await this.ensureInitialized();
      const reviewHistoryStore = await new Promise((resolve, reject) => {
        const tx = this.dbManager.db.transaction("review_history", "readonly");

        const req = tx.objectStore("review_history").getAll();

        req.onsuccess = (e) => resolve(e.target.result);

        req.onerror = (e) => reject(e.target.error);
      });

      return true;
    } catch (error) {
      console.error("Error verifying database persistence:", error);

      throw error;
    }
  }

  // ───────────── INITIALIZE CURRENT CHAPTER ─────────────

  async initializeCurrentChapter() {
    try {
      console.log("🆕 IDB Initializing current chapter...");

      await this.performTransaction(
        ["chapters"],
        "readwrite",
        (tx, resolve, reject) => {
          const chapterStore = tx.objectStore("chapters");

          // PHASE 1: Disabled automatic initial chapter creation
          // Create initial chapter
          // const initialChapter = {
          //   chapter: 0,
          //   title: "Introduction",
          //   is_current: 1,
          // };

          // chapterStore.put(initialChapter);

          tx.oncomplete = () => {
            // PHASE 1: Disabled automatic initial chapter creation
            // this.inMemoryData.currentChapter = initialChapter;
            // this.inMemoryData.currentDeck = [];

            // Initialize chapter sets if modal exists
            if (this.modal) {
              if (!this.modal.chapterSets) {
                this.modal.chapterSets = new Map();
              }
              this.modal.chapterSets.set(0, []);
            }

            // PHASE 1: Disabled automatic initial chapter creation
            // resolve(initialChapter);
            resolve(null);
          };
        }
      );

      return this.inMemoryData.currentChapter;
    } catch (error) {
      console.error("❌ IDB Error initializing current chapter:", error);
      throw error;
    }
  }

  async setCurrentChapter(chapterNumber) {
    try {
      console.log(`🔄 IDB Setting current chapter to ${chapterNumber}...`);

      await this.dbManager.performTransaction(
        "chapters",
        "readwrite",
        async (store) => {
          // Reset all chapters to not current
          const allChapters = await store.getAll();
          for (const chapter of allChapters) {
            chapter.is_current = 0;
            await store.put(chapter);
          }

          // Set new current chapter
          const currentChapter = await store.get(chapterNumber);
          if (currentChapter) {
            currentChapter.is_current = 1;
            await store.put(currentChapter);
            console.log("✅ IDB Updated current chapter:", currentChapter);
          }
        }
      );

      return this.getCurrentChapter();
    } catch (error) {
      console.error("❌ IDB Error setting current chapter:", error);
      throw error;
    }
  }

  async getCurrentChapter() {
    try {
      const currentChapter = await this.dbManager.performTransaction(
        "chapters",
        "readonly",
        async (store) => {
          const chapters = await store.getAll();
          return chapters.find((c) => c.is_current === 1);
        }
      );
      console.log("📖 IDB Got current chapter:", currentChapter);
      return currentChapter || null;
    } catch (error) {
      console.error("❌ IDB Error getting current chapter:", error);
      return null;
    }
  }

  async addChapter(chapter) {
    try {
        await this.ensureInitialized();
        
        // Ensure chapter object has all required fields
        const chapterRecord = {
            chapter: chapter.chapter,
            title: chapter.title || this.getChapterTitle(chapter.chapter),
            is_current: chapter.is_current || 0,
        };
        
        await this.performTransaction("chapters", "readwrite", (tx, resolve, reject) => {
            const store = tx.objectStore("chapters");
            
            // If setting as current, unset others first
            if (chapterRecord.is_current === 1) {
                const getAllReq = store.getAll();
                getAllReq.onsuccess = () => {
                    const chapters = getAllReq.result;
                    chapters.forEach((ch) => {
                        if (ch.chapter !== chapterRecord.chapter) {
                            ch.is_current = 0;
                            store.put(ch);
                        }
                    });
                    // Now save the new chapter
                    const request = store.put(chapterRecord);
                    request.onsuccess = () => {
                        console.log(`✅ IDB Chapter "${chapterRecord.title}" saved successfully.`);
                        resolve();
                    };
                    request.onerror = (event) => {
                        console.error(`❌ IDB Error saving chapter:`, event.target.error);
                        reject(event.target.error);
                    };
                };
                getAllReq.onerror = () => reject(getAllReq.error);
            } else {
                // Just save the chapter
                const request = store.put(chapterRecord);
                request.onsuccess = () => {
                    console.log(`✅ IDB Chapter "${chapterRecord.title}" saved successfully.`);
                    resolve();
                };
                request.onerror = (event) => {
                    console.error(`❌ IDB Error saving chapter:`, event.target.error);
                    reject(event.target.error);
                };
            }
        });
        return true;
    } catch (error) {
        console.error("❌ IDB Error in addChapter:", error);
        // Only fallback if IndexedDB is completely unavailable
        if (this.useFallback && !this.dbManager?.db) {
            console.warn("⚠️ IndexedDB unavailable, falling back to localStorage");
            return super.addChapter(chapter);
        }
        throw error;
    }
  }

  // ───────────── DEBUG TABLES AS CONSOLE TABLE ─────────────

  async debugTablesAsConsoleTable() {
    try {
      console.log("\n📊 === DATABASE TABLES OVERVIEW === 📊");

      for (const storeName of [
        "review_history",
        "chapters",
        "cards",
        "card_tags",
        "current_chapter",
      ]) {
        const data = await new Promise((resolve, reject) => {
          const tx = this.dbManager.db.transaction(storeName, "readonly");

          const req = tx.objectStore(storeName).getAll();

          req.onsuccess = (e) => resolve(e.target.result);

          req.onerror = (e) => reject(e.target.error);
        });

        console.log(`\n📋 Store: ${storeName.toUpperCase()} 📋`);

        if (data && data.length > 0) {
          console.table(data);

          console.log(`Total records: ${data.length}`);
        } else {
          console.log("⚠️ No data in store");
        }

        console.log("----------------------------------------");
      }

      console.log("\n=== END DATABASE OVERVIEW ===\n");
    } catch (error) {
      console.error("Error in debugTablesAsConsoleTable:", error.message);
    }
  }

  // ───────────── SAVE INITIAL DECK AND CARDS ─────────────

  async saveInitialDeckAndCards(chapter, cards) {
    if (this.useFallback) {
      return false;
    }

    try {
      await this.performTransaction(
        ["chapters", "current_chapter", "cards", "card_tags"],
        "readwrite",
        (tx, resolve, reject) => {
          if (chapter) {
            const chaptersStore = tx.objectStore("chapters");

            chaptersStore.put({
              chapter: chapter.chapter,
              title: chapter.title,
            });

            const currentStore = tx.objectStore("current_chapter");

            currentStore.clear();

            currentStore.add({
              chapter: chapter.chapter,
              title: chapter.title,
            });
          }

          const cardStore = tx.objectStore("cards");

          const tagStore = tx.objectStore("card_tags");

          cards.forEach((card) => {
            const cardId = card.id || crypto.randomUUID();

            const now = new Date().toISOString();

            const cardRecord = {
              id: cardId,

              chapter: chapter.chapter,

              question: card.question,

              answer: card.answer,

              created: card.created || now,

              repetitions: card.repetitions || 0,

              easeFactor: card.easeFactor || 2.5,

              interval: card.interval || 0,

              nextReviewDate: card.nextReviewDate || now,

              lastReviewQuality: card.lastReviewQuality || 0,

              tags: card.tags || [],
            };

            cardStore.put(cardRecord);

            if (cardRecord.tags && cardRecord.tags.length > 0) {
              cardRecord.tags.forEach((tag) => {
                tagStore.put({ id: `${cardId}_${tag}`, card_id: cardId, tag });
              });
            }
          });

          tx.oncomplete = () => resolve();
        }
      );

      return true;
    } catch (error) {
      console.error("Error in saveInitialDeckAndCards:", error);

      return false;
    }
  }

  // ───────────── ADD FLASHCARDS FROM TEXT ─────────────

  async addFlashcardsFromText(text) {
    const loadingId = this.notificationHandler.showLoadingState(
      "Processing flashcards..."
    );

    try {
      const currentChapterData = localStorage.getItem("current_chapter");

      if (!currentChapterData) {
        throw new Error("No chapter selected");
      }

      const chapterInfo = JSON.parse(currentChapterData);

      let flashcards;

      const duckAI = await DuckAI.getInstance();

      const flashcardsGenerator = duckAI.generateFlashcards(text);

      console.log("Starting flashcards generation...");

      // Consume the async generator properly
      let finalResult = null;
      let accumulatedResponse = '';
      
      for await (const chunk of flashcardsGenerator) {
        console.log("Received chunk:", chunk);
        
        // If we get a direct array result, use it
        if (Array.isArray(chunk)) {
          finalResult = chunk;
          break;
        }
        
        // Otherwise, accumulate the streaming response
        if (typeof chunk === 'string') {
          accumulatedResponse += chunk;
        }
      }

      // Check if the generator returned a final result
      const generatorResult = await flashcardsGenerator.next();
      if (generatorResult.done && generatorResult.value && Array.isArray(generatorResult.value)) {
        finalResult = generatorResult.value;
      }

      // If we got a final result directly, use it
      if (finalResult) {
        flashcards = finalResult;
        console.log("Using direct result:", flashcards);
      } else {
        // Process the accumulated response
        console.log("Accumulated response:", accumulatedResponse);

        const jsonPattern = /\[([\s\S]*?)\](?=\s*$)/;

        const match = accumulatedResponse.match(jsonPattern);

        if (match) {
          try {
            const jsonStr = `[${match[1]}]`;

            const cleanedJson = jsonStr

              .replace(/\n/g, " ")

              .replace(/\s+/g, " ")

              .replace(/"/g, '"')

              .replace(/"{/g, "{")

              .replace(/}"/g, "}")

              .replace(/\\n/g, "")

              .trim();

            const repairedJson = jsonrepair(cleanedJson);

            flashcards = JSON.parse(repairedJson);
          } catch (e) {
            console.error("Failed to parse JSON:", e);

            throw new Error(`Failed to parse JSON: ${e.message}`);
          }
        } else {
          console.error(
            "No JSON array found in response:",
            accumulatedResponse
          );

          throw new Error("No JSON array found in response");
        }
      }

      if (!Array.isArray(flashcards)) {
        console.error("Parsed content is not an array:", flashcards);

        throw new Error("Parsed content is not an array");
      }

      const result = await this.processFlashcards(
        flashcards,
        chapterInfo,
        loadingId
      );

      return { success: true, flashcards: result };
    } catch (error) {
      console.error("Failed to add flashcards from text:", error);

      this.notificationHandler.updateNotification(
        loadingId,
        "Failed to add flashcards",
        "error"
      );

      throw error;
    }
  }

  // ───────────── PROCESS FLASHCARDS ─────────────

  async processFlashcards(flashcards, chapterInfo, loadingId) {
    return await this.performTransaction(
      ["cards", "card_tags"],
      "readwrite",
      (tx, resolve, reject) => {
        const cardStore = tx.objectStore("cards");

        const tagStore = tx.objectStore("card_tags");

        const processedCards = [];

        flashcards.forEach((card) => {
          const processedCard = {
            ...card,

            id: crypto.randomUUID(),

            tags: ["AI-Generated"],

            created: new Date().toISOString(),

            repetitions: 0,

            easeFactor: 2.5,

            interval: 0,

            nextReviewDate: null,

            reviewHistory: [],
          };

          cardStore.add(processedCard);

          tagStore.add({
            id: `${processedCard.id}_AI-Generated`,
            card_id: processedCard.id,
            tag: "AI-Generated",
          });

          processedCards.push(processedCard);
        });

        tx.oncomplete = () => {
          this.inMemoryData.currentDeck = [
            ...this.inMemoryData.currentDeck,
            ...processedCards,
          ];

          if (this.modal) {
            if (!this.modal.flashcards) {
              this.modal.flashcards = [];
            }

            this.modal.flashcards = [
              ...this.modal.flashcards,
              ...processedCards,
            ];

            if (!this.modal.chapterSets.has(chapterInfo.chapter)) {
              this.modal.chapterSets.set(chapterInfo.chapter, []);
            }

            const chapterCards = this.modal.chapterSets.get(
              chapterInfo.chapter
            );

            this.modal.chapterSets.set(chapterInfo.chapter, [
              ...chapterCards,
              ...processedCards,
            ]);

            requestAnimationFrame(() => {
              this.modal.renderChapters && this.modal.renderChapters();

              this.modal.renderTags && this.modal.renderTags();

              this.modal.uiHandler &&
                this.modal.uiHandler.updateProgress &&
                this.modal.uiHandler.updateProgress();
            });
          }

          this.notificationHandler.updateNotification(
            loadingId,
            "Cards added successfully!",
            "success"
          );

          resolve(processedCards);
        };
      }
    );
  }

  // ───────────── DEBOUNCE HELPER ─────────────

  debounce(func, wait) {
    let timeout;

    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);

        func(...args);
      };

      clearTimeout(timeout);

      timeout = setTimeout(later, wait);
    };
  }

  // (Placeholder for getChapterTitle; implement as needed.)

  getChapterTitle(chapter) {
    return `Chapter ${chapter}`;
  }

  async addReviewActivity(date = new Date()) {
    try {
      await this.ensureInitialized();

      const dateStr = date.toISOString().split("T")[0];

      // Check if we already have an entry for this date
      const existingEntries = await this.dbManager.performTransaction(
        "review_history",
        "readonly",
        (store) => {
          return store.get(dateStr);
        }
      );

      if (existingEntries) {
        // Update existing entry
        await this.dbManager.performTransaction(
          "review_history",
          "readwrite",
          (store) => {
            return store.put({
              date: dateStr,
              count: existingEntries.count + 1,
            });
          }
        );
      } else {
        // Create new entry
        await this.dbManager.performTransaction(
          "review_history",
          "readwrite",
          (store) => {
            return store.add({
              date: dateStr,
              count: 1,
            });
          }
        );
      }

      // Update in-memory data
      if (!this.inMemoryData.reviewHistory[dateStr]) {
        this.inMemoryData.reviewHistory[dateStr] = 0;
      }
      this.inMemoryData.reviewHistory[dateStr]++;

      return true;
    } catch (error) {
      console.error("Error updating review activity:", error);
      if (this.useFallback) {
        return super.addReviewActivity(date);
      }
      return false;
    }
  }

  async createInitialDeck() {
    if (this.initialDeck) {
      return null;
    }
    this.initialDeck = true;

    try {
      // Create initial chapter data
      const currentChapter = {
        chapter: 0,
        title: "Introduction",
        is_current: 1,
      };

      // await this.dbManager.performTransaction(
      //   'chapters',
      //   'readwrite',
      //   (store) => {
      //     return store.put(currentChapter);
      //   }
      // );

      // Create the initial card
      const card = {
        id: crypto.randomUUID(),
        chapter: 0,
        question: "👋 How to Use Spaced Repetition App",
        answer: `# How to Use the Spaced Repetition App

This app helps you learn material by creating flashcards and tracking your progress.

## Key Features

* **Decks:** Each chapter of your textbook can be a separate deck of cards.
* **Cards:**  Each card contains a question or concept to be learned.
* **Progress Tracking**: See total cards, mastered cards and cards needing review.
* **Learning Progress Graph**: See a graph of your learned cards.
* **Review Cards**:  Practice your cards based on a spaced repetition algorithm.
* **Add New Cards:** Create new flashcards to help with memorization.
* **Full Markdown Editor**: Use a full markdown editor to create your cards.
* **Tagging**: Tag your cards to help with organization and filtering by using # : \`#tags\`

## Basic Steps

1. **Create Decks:** The application will create a deck for each chapter, which should happen automatically when you visit. 
2. **Add Cards:** As you study, add a new card for each important fact, concept, or question.
3. **Review Cards:** Periodically review the cards in each deck, making sure to keep up with new cards.
4. **Track Progress**: Look at the learning progress to see what cards are mastered and which are not.
5. **Track Quality**: Look at quality graphs to see how well you are doing in rating your reviews.

#introduction`,
        created: new Date().toISOString(),
        repetitions: 0,
        easeFactor: 2.5,
        interval: 0,
        nextReviewDate: new Date().toISOString(),
        lastReviewQuality: 0,
        chapter_title: "Introduction",
        tags: ["introduction"],
        reviewHistory: [],
      };

      // Save to database
      // await Promise.all([
      //     this.dbManager.performTransaction(
      //         'chapters',
      //         'readwrite',
      //         (store) => store.put(currentChapter)
      //     ),
      //     this.dbManager.performTransaction(
      //         'cards',
      //         'readwrite',
      //         (store) => store.put(card)
      //     ),
      //     this.dbManager.performTransaction(
      //         'card_tags',
      //         'readwrite',
      //         (store) => store.put({
      //             id: `${card.id}_introduction`,
      //             card_id: card.id,
      //             tag: 'introduction'
      //         })
      //     )
      // ]);

      // Update in-memory data
      this.inMemoryData.currentChapter = currentChapter;
      this.inMemoryData.currentDeck = [card];
      this.inMemoryData.allTags = new Set(["introduction"]);
      this.inMemoryData.reviewHistory = {
        [card.created.split("T")[0]]: 1,
      };

      // Create the return dataset
      const chapterSets = new Map();
      chapterSets.set(0, [card]);

      // SIDE EFFECT EFFECT EFFECT
      this.currentChapter = currentChapter;

      const initialDataset = {
        currentChapter: {
          chapter: currentChapter.chapter,
          title: currentChapter.title,
        },
        chapterSets: chapterSets,
        flashcards: [card],
        allTags: Array.from(this.inMemoryData.allTags),
        reviewHistory: this.inMemoryData.reviewHistory,
      };

      // Handle database updates asynchronously
      this.updateDatabase(currentChapter, card).catch((error) => {
        console.error("Error in background database update:", error);
      });

      console.log("✅ IDB Initial deck dataset prepared", initialDataset);
      return initialDataset;
    } catch (error) {
      console.error("Error creating initial deck:", error);
      return null;
    }
  }

  // New method to handle database updates in the background
  async updateDatabase(currentChapter, card) {
    try {
      await this.ensureInitialized();

      await Promise.all([
        this.dbManager.performTransaction("chapters", "readwrite", (store) =>
          store.put(currentChapter)
        ),
        this.dbManager.performTransaction("cards", "readwrite", (store) =>
          store.put(card)
        ),
        this.dbManager.performTransaction("card_tags", "readwrite", (store) =>
          store.put({
            id: `${card.id}_introduction`,
            card_id: card.id,
            tag: "introduction",
          })
        ),
      ]);
      console.log("✅ IDB Background database update completed");
    } catch (error) {
      console.error("❌ IDB Background database update failed:", error);
      // Could implement retry logic here if needed
    }
  }

  async loadFromLocalStorage(isNoMessage = false) {
    try {
      await this.ensureInitialized();

      // First check localStorage for current_chapter
      const localStorageChapter = this.getLocalStorageChapter();

      // Get current chapter with fallback
      let currentChapter;
      try {
        currentChapter = await this.dbManager.performTransaction(
          "chapters",
          "readwrite",
          (store) => store.getAll()
        );
      } catch (chapterError) {
        console.error("❌ IDB Error loading chapters:", chapterError);
        throw chapterError;
      }

      // KAI NEEDS TO FIX THIS HOW DOES SQLITE GET CURRENT CHAPTER?
      // If no chapters exist, create initial deck or use localStorage chapter
      if (currentChapter.length === 0) {
        // if (localStorageChapter) {
        //     console.log("📦 IDB Using chapter from localStorage:", localStorageChapter);
        //     currentChapter = [localStorageChapter];
        // } else {
        // PHASE 1: Disabled automatic initial deck creation
        // const initialData = await this.createInitialDeck();
        // if (initialData) {
        //   console.trace(
        //     "🎉 IDB Initial deck created successfully",
        //     initialData
        //   );
        //   return initialData;
        // }
        // }
      }
      // PHASE 1: Removed line that was overriding database chapters with localStorage chapter
      // currentChapter = [localStorageChapter];

      let current = currentChapter.find((c) => c.is_current === 1);

      // If no current chapter found, use localStorage or default to Introduction
      if (!current && localStorageChapter) {
        current = localStorageChapter;
        console.log("📦 IDB Using localStorage chapter as current:", current);
      }

      !current &&
        console.warn(
          "🚨 IDB No current chapter found, defaulting to Introduction..."
        );

      console.log("🔄 IDB Current chapter:", currentChapter, current);
      currentChapter = {
        chapter: current?.chapter || 0,
        title: current?.title || "Introduction",
      };

      this.inMemoryData.currentChapter = currentChapter || null;

      // Load all cards
      let cards;
      try {
        cards = await this.dbManager.performTransaction(
          "cards",
          "readonly",
          (store) => store.getAll()
        );
      } catch (cardsError) {
        console.error("❌ IDB Error loading cards:", cardsError);
        throw cardsError;
      }

      // Load all tags
      const tags = await this.dbManager.performTransaction(
        "card_tags",
        "readonly",
        (store) => {
          return store.getAll();
        }
      );

      // Process cards and their tags
      const chapterMap = new Map();

      cards.forEach((card) => {
        if (!chapterMap.has(card.chapter)) {
          chapterMap.set(card.chapter, []);
        }

        // Get tags for this card
        const cardTags = tags
          .filter((t) => t.card_id === card.id)
          .map((t) => t.tag);

        const processedCard = {
          ...card,
          tags: card.tags || [],
          nextReviewDate: card.nextReviewDate
            ? new Date(card.nextReviewDate)
            : null,
        };
        chapterMap.get(card.chapter).push(processedCard);

        // Update current deck if this is the current chapter
        if (
          this.inMemoryData.currentChapter &&
          card.chapter === this.inMemoryData.currentChapter.chapter
        ) {
          this.inMemoryData.currentDeck = chapterMap.get(card.chapter);
        }

        // Collect tags
        cardTags.forEach((tag) => {
          this.inMemoryData.allTags.add(tag);
        });
      });

      // Load review history
      const reviewHistory = await this.dbManager.performTransaction(
        "review_history",
        "readonly",
        (store) => {
          return store.getAll();
        }
      );

      this.inMemoryData.reviewHistory = reviewHistory.reduce((acc, review) => {
        acc[review.date] = review.count;
        return acc;
      }, {});

      console.log("🔄 IDB Returning data to modal:", this.inMemoryData);
      const returnData = {
        currentChapter: currentChapter,
        chapterSets: chapterMap, // We'll populate this
        flashcards: cards || [],
        allTags: Array.from(this.inMemoryData.allTags),
        reviewHistory: this.inMemoryData.reviewHistory,
      };

      return returnData;
    } catch (error) {
      console.error("❌ IDB Error in loadFromLocalStorage:", error);
      console.error("❌ IDB Error stack:", error.stack);
      if (this.useFallback) {
        return super.loadFromLocalStorage();
      }
      return {
        currentChapter: null,
        chapterSets: new Map(),
        flashcards: [],
        allTags: [],
        reviewHistory: {},
      };
    }
  }

  // Add this helper method to get chapter from localStorage
  getLocalStorageChapter() {
    try {
      const currentChapterData = localStorage.getItem("current_chapter");
      if (!currentChapterData) return null;

      const chapter = JSON.parse(currentChapterData);
      return {
        chapter: chapter.chapter,
        title: chapter.title,
        is_current: 1,
      };
    } catch (error) {
      console.error("❌ IDB Error reading from localStorage:", error);
      return null;
    }
  }
}
