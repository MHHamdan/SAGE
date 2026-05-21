"""Tests for skill registry."""

import pytest


class TestSkillRegistry:
    """Test skill registry functionality."""

    def test_register_skill(self):
        """Test skill registration."""
        from sage.skills import Skill, SkillRegistry

        registry = SkillRegistry()

        def test_func(x):
            return x * 2

        skill = Skill(
            name="doubler",
            description="Doubles a number",
            function=test_func,
        )

        registry.register(skill)
        assert "doubler" in registry.list_skills()

    def test_get_skill(self):
        """Test skill retrieval."""
        from sage.skills import Skill, SkillRegistry

        registry = SkillRegistry()

        def test_func(x):
            return x * 2

        skill = Skill(name="test", description="Test skill", function=test_func)
        registry.register(skill)

        retrieved = registry.get("test")
        assert retrieved is not None
        assert retrieved.name == "test"

    def test_unregister_skill(self):
        """Test skill unregistration."""
        from sage.skills import Skill, SkillRegistry

        registry = SkillRegistry()

        def test_func():
            pass

        skill = Skill(name="temp", description="Temporary", function=test_func)
        registry.register(skill)
        registry.unregister("temp")

        assert "temp" not in registry.list_skills()

    def test_search_skills(self):
        """Test skill search by query."""
        from sage.skills import Skill, SkillRegistry

        registry = SkillRegistry()

        skills = [
            Skill(
                name="search_web", description="Search the web", function=lambda: None
            ),
            Skill(name="read_file", description="Read a file", function=lambda: None),
            Skill(
                name="write_file", description="Write to a file", function=lambda: None
            ),
        ]

        for skill in skills:
            registry.register(skill)

        # Search for file-related skills
        results = registry.search("file")
        assert len(results) >= 2


class TestSkill:
    """Test Skill class."""

    def test_skill_creation(self):
        """Test skill creation."""
        from sage.skills import Skill

        def my_func(x, y):
            return x + y

        skill = Skill(
            name="adder",
            description="Adds two numbers",
            function=my_func,
            version="1.0.0",
            trust_score=0.9,
        )

        assert skill.name == "adder"
        assert skill.version == "1.0.0"
        assert skill.trust_score == 0.9

    def test_skill_execution(self):
        """Test skill execution."""
        from sage.skills import Skill

        def multiply(a, b):
            return a * b

        skill = Skill(name="multiply", description="Multiply", function=multiply)
        result = skill.execute(a=3, b=4)

        assert result == 12

    def test_skill_stats_update(self):
        """Test skill statistics update."""
        from sage.skills import Skill

        def test_func():
            return "done"

        skill = Skill(name="test", description="Test", function=test_func)
        skill.execute()
        skill.update_stats(success=True, duration=1.5)

        assert skill.total_calls == 1
        assert skill.successes == 1
