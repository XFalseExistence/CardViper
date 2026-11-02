package com.cardviper.app.ui.imagetest

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SessionADemoDisclosureTest {
    @Test fun demoDisclosureStatesFaceUpOnlyAndOffline() {
        val disclosure = ImageTestDisclosure.forBuild(isSessionADemo = true).joinToString(" ")

        assertTrue(disclosure.contains("SESSION A"))
        assertTrue(disclosure.contains("FACE-UP"))
        assertTrue(disclosure.contains("OFFLINE"))
        assertTrue(disclosure.contains("never change your active shoe"))
        assertTrue(disclosure.contains("No quality claim"))
    }

    @Test fun standardDisclosureDoesNotClaimTheDemoIsInstalled() {
        val disclosure = ImageTestDisclosure.forBuild(isSessionADemo = false).joinToString(" ")

        assertTrue(disclosure.contains("not active"))
        assertFalse(disclosure.contains("SESSION A / FACE-UP DEMO"))
    }
}
