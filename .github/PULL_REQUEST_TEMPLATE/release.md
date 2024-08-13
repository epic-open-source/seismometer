<!-- 
1. Start creating a PR normally
2. Append the URL with ?template=release.md
3. Hit the button to create (again) and the GUI will be populated with the following template
-->
## Pull Request
* [ ] Verify all tests are passing
* [ ] Update the version in seismometer._version.py
* [ ] Call towncrier to draft the update to the release notes.  Delete fragments as needed.
<!--  Do any manual release testing and review here -->
* [ ] Work with the Core team to get approvals
* [ ] Squash merge the release pull request with message "`Release <VERSION>`"

## Post Merge
* [ ] Tag the commit with the matching version vX.Y.Z
* [ ] Create a release from this tag
* [ ] If done prior to a deploy workflow, work with the Core team to publish to PyPI.
* [ ] Make announcement.
