package com.cardviper.app.blackjack

import com.cardviper.app.model.CardLedgerEvent
import com.cardviper.app.model.LedgerEventType

class LedgerResolver {
    fun resolve(events: List<CardLedgerEvent>): List<ResolvedCard> {
        val rootForEvent = mutableMapOf<String, String>()
        val effective = linkedMapOf<String, ResolvedCard>()

        events.sortedWith(compareBy<CardLedgerEvent> { it.sequenceNumber }.thenBy { it.eventId })
            .forEach { event ->
                when (event.eventType) {
                    LedgerEventType.CARD_COMMITTED,
                    LedgerEventType.CARD_MANUAL_ADDED -> {
                        val card = event.card ?: return@forEach
                        if (rootForEvent.containsKey(event.eventId)) return@forEach
                        rootForEvent[event.eventId] = event.eventId
                        effective[event.eventId] = ResolvedCard(
                            rootEventId = event.eventId,
                            effectiveEventId = event.eventId,
                            card = card,
                            source = event.source,
                        )
                    }

                    LedgerEventType.CARD_CORRECTED -> {
                        val target = event.targetEventId ?: return@forEach
                        val root = rootForEvent[target] ?: return@forEach
                        val card = event.card ?: return@forEach
                        val existing = effective[root] ?: return@forEach
                        rootForEvent[event.eventId] = root
                        effective[root] = existing.copy(
                            effectiveEventId = event.eventId,
                            card = card,
                            source = event.source,
                        )
                    }

                    LedgerEventType.CARD_INVALIDATED -> {
                        val target = event.targetEventId ?: return@forEach
                        val root = rootForEvent[target] ?: return@forEach
                        rootForEvent[event.eventId] = root
                        effective.remove(root)
                    }
                }
            }

        return effective.values.toList()
    }
}
