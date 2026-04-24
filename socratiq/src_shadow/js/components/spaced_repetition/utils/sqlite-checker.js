let cachedStorageType = null;

export async function checkStorageSupport() {
  if (cachedStorageType) {
    return cachedStorageType;
  }

  // FORCE IndexedDB
  console.log("⚠️ SQLite disabled - using IndexedDB for persistence");
  cachedStorageType = "indexeddb";
  return "indexeddb";
}

// Add storage migration utility
export async function migrateStorage(from, to, notification) {
  try {
    notification.showLoadingState(
      "Migrating data to more compatible storage...",
    );

    // Get data from old storage
    const oldData = await from.exportToJSON();

    // Import to new storage
    await to.importFromJSON(oldData);

    notification.updateNotification(
      null,
      "Storage upgraded successfully!",
      "success",
    );

    return true;
  } catch (error) {
    console.error("Migration failed:", error);
    notification.updateNotification(
      null,
      "Storage upgrade failed. Some features may be limited.",
      "error",
    );
    return false;
  }
}
